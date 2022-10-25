# movie-recommender-system

Group assignment for the Information Retrieval course at BITS Pilani, Hyderabad

## Usage

`python cf.py` : Returns the RMSE and MAE loss metrics on test data using three different approaches of collaborative filtering namely:

* user-user filtering
* item-item filtering
* baseline approach

`python svd.py` : Performs Singular Value Decomposition on the given utility matrix and report the reconstruction error (RMSE and MAE loss) at the specified energy.

`python cur.py` : Performs CUR decomposition and reports reconstruction error for the specified r value. r is the parameter which specifies the number of columns and rows in C and R matrix respectively in CUR.

`python main.py` : Latent Factor model implemented using Stochastic Gradient Descent - learns the latent (hidden) factors for each user and movie.

## Results

### Latent Factor model (learning rate = 0.01, regularisation coefficient = 0.05, epochs = 50
| Latent Factors | RMSE | MAE |
| -------------- | ---- | --- |
| 10 |	0.836 |	0.654 |
| 20 |	0.840 |	0.657 |
| 50 |	0.829 |	0.649 |
| 100 |	0.833 |	0.653 |

[Loss curve for LF](plots/loss.png)

### Collaborative Filtering
| Approach | RMSE | MAE |
| -------------- | ---- | --- |
| Baseline |	0.904 |	0.724 |
| user-user filtering |	1.147 |	0.831 |
| item-item filtering |	0.921|	0.730|

### SVD
| Energy | RMSE | MAE |
| -------------- | ---- | --- |
| 90 |	0.243 |	0.132 |

### CUR Decomposition
| r | RMSE | MAE |
| -------------- | ---- | --- |
| 3000 |	0.615 |	0.205 |
| 2000 | 2.109 | 0.276 |
