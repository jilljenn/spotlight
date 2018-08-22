"""
Models for recommending items given a sequence of previous items
a user has interacted with.
"""

import numpy as np

import torch

import torch.optim as optim

from spotlight.helpers import _repr_model
from spotlight.losses import (adaptive_hinge_loss,
                              bpr_loss,
                              hinge_loss,
                              pointwise_loss, regression_loss, squared_errors)
from spotlight.sequence.representations import (PADDING_IDX, CNNNet,
                                                LSTMNet,
                                                MixtureLSTMNet,
                                                PoolNet)
from spotlight.sampling import sample_items
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
from spotlight.layers import ScaledEmbedding


class ExplicitSequenceModel(object):
    """
    Model for sequential recommendations using implicit feedback.

    Parameters
    ----------

    loss: string, optional
        The loss function for approximating a softmax with negative sampling.
        One of 'pointwise', 'bpr', 'hinge', 'adaptive_hinge', corresponding
        to losses from :class:`spotlight.losses`.
    representation: string or instance of :class:`spotlight.sequence.representations`, optional
        Sequence representation to use. If string, it must be one
        of 'pooling', 'cnn', 'lstm', 'mixture'; otherwise must be one of the
        representations from :class:`spotlight.sequence.representations`
    embedding_dim: int, optional
        Number of embedding dimensions to use for representing items.
        Overridden if representation is an instance of a representation class.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Minibatch size.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a PyTorch optimizer. Overrides l2 and learning
        rate if supplied. If no optimizer supplied, then use ADAM by default.
    use_cuda: boolean, optional
        Run the model on a GPU.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    num_negative_samples: int, optional
        Number of negative samples to generate for adaptive hinge loss.

    Notes
    -----

    During fitting, the model computes the loss for each timestep of the
    supplied sequence. For example, suppose the following sequences are
    passed to the ``fit`` function:

    .. code-block:: python

       [[1, 2, 3, 4, 5],
        [0, 0, 7, 1, 4]]


    In this case, the loss for the first example will be the mean loss
    of trying to predict ``2`` from ``[1]``, ``3`` from ``[1, 2]``,
    ``4`` from ``[1, 2, 3]`` and so on. This means that explicit padding
    of all subsequences is not necessary (although it is possible by using
    the ``step_size`` parameter of
    :func:`spotlight.interactions.Interactions.to_sequence`.
    """

    def __init__(self,
                 loss='pointwise',
                 representation='pooling',
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer_func=None,
                 use_cuda=False,
                 sparse=False,
                 random_state=None,
                 num_negative_samples=5):

        assert loss in ('pointwise',
                        'bpr',
                        'hinge',
                        'adaptive_hinge')

        if isinstance(representation, str):
            assert representation in ('pooling',
                                      'cnn',
                                      'lstm',
                                      'lstm+',
                                      'mixture')

        self._loss = loss
        self._representation = representation
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._optimizer_func = optimizer_func
        self._random_state = random_state or np.random.RandomState()
        self._num_negative_samples = num_negative_samples

        self._num_items = None
        self._net = None
        self._optimizer = None
        self._loss_func = None

        set_seed(self._random_state.randint(-10**8, 10**8),
                 cuda=self._use_cuda)

    def __repr__(self):

        return _repr_model(self)

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):

        self._num_items = interactions.num_items
        self.item_embedding = ScaledEmbedding(self._num_items, self._embedding_dim, sparse=False)

        if self._representation == 'pooling':
            self._net = PoolNet(self._num_items,
                                self._embedding_dim,
                                sparse=self._sparse)
        elif self._representation == 'cnn':
            self._net = CNNNet(self._num_items,
                               self._embedding_dim,
                               sparse=self._sparse)
        elif self._representation == 'lstm':
            self._net = LSTMNet(self._num_items,
                                self._embedding_dim,
                                sparse=self._sparse)
        elif self._representation == 'lstm+':
            self._net = LSTMNet(2 * self._num_items,
                                self._embedding_dim,
                                sparse=self._sparse)
        elif self._representation == 'mixture':
            self._net = MixtureLSTMNet(self._num_items,
                                       self._embedding_dim,
                                       sparse=self._sparse)
        else:
            self._net = self._representation

        self._net = gpu(self._net, self._use_cuda)

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                list(self._net.parameters()) + list(self.item_embedding.parameters()),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer_func(self._net.parameters())

        if self._loss == 'pointwise':
            self._loss_func = pointwise_loss
        elif self._loss == 'bpr':
            self._loss_func = bpr_loss
        elif self._loss == 'hinge':
            self._loss_func = hinge_loss
        else:
            self._loss_func = adaptive_hinge_loss

    def _check_input(self, item_ids):

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def fit(self, interactions, ratings, verbose=False):
        """
        Fit the model.

        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------

        interactions: :class:`spotlight.interactions.SequenceInteractions`
            The input sequence dataset.
        """

        sequences = interactions.sequences.astype(np.int64)

        if not self._initialized:
            self._initialize(interactions)

        self._check_input(sequences)

        for epoch_num in range(self._n_iter):

            indices = np.arange(len(sequences))
            indices = shuffle(indices,
                                random_state=self._random_state)

            sequences_tensor = gpu(torch.from_numpy(sequences),
                                   self._use_cuda)

            epoch_loss = 0.0

            for minibatch_num, batch_indices in enumerate(minibatch(indices,
                                                                    batch_size=self._batch_size)):

                sequence_var = sequences_tensor[batch_indices]
                # print('seq', sequence_var, sequence_var.shape)
                # print('ext', sequence_var[0].numpy())
                interesting = np.where(sequence_var != PADDING_IDX)
                # print(interesting)
                sequence_var = sequence_var[interesting].reshape(1, -1)
                # print('seq', sequence_var, sequence_var.shape)

                item_batch = sequence_var[:, 1:]
                user_batch = interactions.user_ids[batch_indices]
                # print('user batch', user_batch)

                # print('pred', pred)

                truth = np.array([ratings[u, i] for u, i in np.broadcast(user_batch[:, None], sequence_var)]).reshape(self._batch_size, -1)
                truth = torch.from_numpy(truth)#.detach_()
                # print('truth', truth.shape)
                # print('truth', truth)

                pred = self.item2pred(sequence_var, truth)
                # print('pred', pred.shape)

                self._optimizer.zero_grad()
                # print('pos', positive_prediction)
                # print('neg', negative_prediction)

                mask = item_batch != PADDING_IDX
                # print(mask.shape)
                # print(mask)
                real_loss, nb_ent = squared_errors(truth[:, 1:], pred, mask=mask)
                real_loss /= nb_ent
                # print('wow', real_loss)
                epoch_loss += real_loss.item()

                real_loss.backward()

                self._optimizer.step()

                # break

            epoch_loss /= minibatch_num + 1

            if verbose:
                # print('omg')
                print('batches', minibatch_num)
                print('Epoch {}: loss {} '.format(epoch_num, epoch_loss))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'
                                 .format(epoch_loss))

    def _get_negative_prediction(self, shape, user_representation):

        negative_items = sample_items(
            self._num_items,
            shape,
            random_state=self._random_state)
        negative_var = gpu(torch.from_numpy(negative_items), self._use_cuda)

        negative_prediction = self._net(user_representation, negative_var)

        return negative_prediction

    def _get_multiple_negative_predictions(self, shape, user_representation,
                                           n=5):
        batch_size, sliding_window = shape
        size = (n,) + (1,) * (user_representation.dim() - 1)
        negative_prediction = self._get_negative_prediction(
            (n * batch_size, sliding_window),
            user_representation.repeat(*size))

        return negative_prediction.view(n, batch_size, sliding_window)

    def item2pred(self, sequence_var, truth):

        item_batch = sequence_var[:, 1:]
        if self._representation == 'lstm+':
            seq_with_ratings = 2 * sequence_var + (truth >= 3).long()
            # print(seq_with_ratings)
        else:
            seq_with_ratings = sequence_var
        user_representations, last_user_representation = self._net.user_representation(seq_with_ratings)
        item_representation = self.item_embedding(item_batch).transpose(1, 2)
        pred = (user_representations[:, :, 1:] * item_representation).sum(1)

        return pred

    def predict(self, sequences, truth, item_ids=None):
        """
        Make predictions: given a sequence of interactions, predict
        the next item in the sequence.

        Parameters
        ----------

        sequences: array, (1 x max_sequence_length)
            Array containing the indices of the items in the sequence.
        item_ids: array (num_items x 1), optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------

        predictions: array
            Predicted scores for all items in item_ids.
        """

        self._net.train(False)

        sequences = np.atleast_2d(sequences)

        if item_ids is None:
            item_ids = np.arange(self._num_items).reshape(-1, 1)

        # self._check_input(item_ids)
        self._check_input(sequences)

        truth = torch.from_numpy(truth)
        sequences = torch.from_numpy(sequences.astype(np.int64).reshape(1, -1))
        # item_ids = torch.from_numpy(item_ids.astype(np.int64))

        sequence_var = gpu(sequences, self._use_cuda)
        # item_var = gpu(item_ids, self._use_cuda)

        pred = self.item2pred(sequence_var, truth)

        return cpu(pred).detach().numpy().flatten()
