from spotlight.cross_validation import user_based_train_test_split, random_train_test_split
from spotlight.datasets.synthetic import generate_sequential
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import sequence_mrr_score, sequence_rmse_score, rmse_score
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.explicit import ExplicitSequenceModel
from spotlight.factorization.explicit import ExplicitFactorizationModel
from collections import Counter
import pandas as pd
import numpy as np
import pickle

LOAD_EXISTING = True
SEQ_LEN = 30

# Import data
# dataset = generate_sequential(num_users=5, num_items=15, num_interactions=30, concentration_parameter=0.01, order=3)
dataset = get_movielens_dataset(variant='100K')

# Prepare data
if not LOAD_EXISTING:
    nb_ratings_per_user = Counter(dataset.user_ids)
    arr = np.column_stack((dataset.user_ids, dataset.item_ids, dataset.ratings, dataset.timestamps))
    df = pd.DataFrame(arr, columns=('user', 'item', 'rating', 'time'))

    kept_users = list(filter(lambda user_id: nb_ratings_per_user[user_id] <= SEQ_LEN, np.unique(dataset.user_ids)))
    print(len(kept_users), 'users')

    lite = df.query("user in @kept_users")
    print(lite.shape, 'ratings')

    with open('notebooks/lite.pickle', 'wb') as f:
        pickle.dump(lite, f)
else:
    with open('notebooks/lite.pickle', 'rb') as f:
        lite = pickle.load(f)

from spotlight.interactions import Interactions

user_ids = np.array(lite['user']).astype(np.int32)
item_ids = np.array(lite['item']).astype(np.int32)
ratings = np.array(lite['rating']).astype(np.float32)
times = np.array(lite['time']).astype(np.int32)
dataset = Interactions(user_ids, item_ids, ratings, times)

# Prepare train test
train, test = user_based_train_test_split(dataset)
# train, test = random_train_test_split(dataset)

# Test baseline
model = ExplicitFactorizationModel(n_iter=20)
model.fit(train, verbose=True)
print('RMSE', rmse_score(model, test))

from scipy.sparse import coo_matrix

ratings = coo_matrix((dataset.ratings, (dataset.user_ids, dataset.item_ids)), shape=(dataset.num_users, dataset.num_items)).tocsr()

train_seq = train.to_sequence(SEQ_LEN)
test_seq = test.to_sequence(SEQ_LEN)

model = ExplicitSequenceModel(n_iter=30, representation='lstm', batch_size=1)
model.fit(train_seq, ratings, verbose=True)

SEQ_ID = 0
user_batch = train_seq.user_ids[SEQ_ID]
item_batch = train_seq.sequences[SEQ_ID]
print('seq', item_batch)
item_batch = np.trim_zeros(item_batch)

truth = np.array([ratings[u, i] for u, i in np.broadcast(user_batch, item_batch)]).reshape(1, -1)
pred = model.predict(item_batch, truth)
print(pred)
print(truth)

user_batch = test_seq.user_ids[SEQ_ID]
item_batch = test_seq.sequences[SEQ_ID]
print('seq', item_batch)
item_batch = np.trim_zeros(item_batch)

truth = np.array([ratings[u, i] for u, i in np.broadcast(user_batch, item_batch)]).reshape(1, -1)
pred = model.predict(item_batch, truth)
print(pred)
print(truth)

print('RMSE', sequence_rmse_score(model, test_seq, ratings))

# train, test = user_based_train_test_split(dataset)

# train = train.to_sequence()
# print(train.__dict__)
# test = test.to_sequence()

# model = ImplicitSequenceModel(n_iter=3,
#                               representation='lstm',
#                               loss='bpr')
# model.fit(train)

# mrr = sequence_mrr_score(model, test)
# print(mrr.mean())
