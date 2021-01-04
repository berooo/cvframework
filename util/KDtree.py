#implementation of KD-tree
from random import randint
import numpy as np
def min_max_scale(x):
  """Scale the element of X into an interval [0, 1].
      Arguments:
          X {ndarray} -- 2d array object with int or float
      Returns:
          ndarray -- 2d array object with float
      """
  x_max = x.max(axis=0)
  x_min = x.min(axis=0)

  return (x - x_min) / (x_max - x_min)


def get_euclidean_distance(arr1, arr2):
  """"Calculate the Euclidean distance of two vectors.
  Arguments:
      arr1 {ndarray}
      arr2 {ndarray}
  Returns:
      float
  """
  return ((arr1 - arr2) ** 2).sum() ** 0.5


def gen_data(low, high, n_rows, n_cols=None):
  """Generate dataset randomly.
  Arguments:
      low {int} -- The minimum value of element generated.
      high {int} -- The maximum value of element generated.
      n_rows {int} -- Number of rows.
      n_cols {int} -- Number of columns.
  Returns:
      list -- 1d or 2d list with int
  """
  if n_cols is None:
    ret = [randint(low, high) for _ in range(n_rows)]
  else:
    ret = [[randint(low, high) for _ in range(n_cols)]
           for _ in range(n_rows)]
  return ret
class Node(object):
  def __init__(self):
    self.father=None
    self.left=None
    self.right=None
    self.feature=None
    self.split=None

  def __str__(self):
    return "feature: %s, split: %s" % (str(self.feature), str(self.split))

  @property
  def brother(self):
    if self.father is None:
      ret=None
    else:
      if self.father.left is self:
        ret=self.father.right
      else:
        ret=self.father.left
    return ret


class KDTree(object):
  def __init__(self):
    self.root=Node()

  def __str__(self):
    ret = []
    i = 0
    que = [(self.root, -1)]
    while que:
      nd, idx_father = que.pop(0)
      ret.append("%d -> %d: %s" % (idx_father, i, str(nd)))
      if nd.left is not None:
        que.append((nd.left, i))
      if nd.right is not None:
        que.append((nd.right, i))
      i += 1
    return "\n".join(ret)

  def __get_median_idx(self,X,idxs,feature):
    n=len(idxs)
    k=n//2
    col=map(lambda i:(i,X[i][feature]),idxs)
    sorted_idxs=map(lambda x:x[0],sorted(col,key=lambda x:x[1]))
    median_idx=list(sorted_idxs)[k]
    return median_idx

  def __get_variance(self,X,idxs,features):
    n=len(idxs)
    col_sum=col_sum_sqr=0
    for idx in idxs:
      xi=X[idx][features]
      col_sum+=xi
      col_sum_sqr+=xi**2
    return col_sum_sqr/n-(col_sum/n)**2

  def __choose_feature(self,X,idxs):
    #选择方差最大的一维
    m=len(X[0])
    variance=map(lambda j:(
      j,self.__get_variance(X,idxs,j)),range(m))
    return max(variance,key=lambda x:x[1])[0]

  def _split_feature(self,X,idxs,features,median_idx):
    idxs_split=[[],[]]
    split_val=X[median_idx][features]
    for idx in idxs:
      if idx==median_idx:
        continue
      xi=X[idx][features]
      if xi<split_val:
        idxs_split[0].append(idx)
      else:
        idxs_split[1].append(idx)
    return idxs_split

  def buildTree(self,X,y):
    X_scale=min_max_scale(np.array(X))
    nd=self.root
    idxs=range(len(X))
    que=[(nd,idxs)]
    while que:
      nd,idxs=que.pop(0)
      n=len(idxs)
      if n==1:
        nd.split=(X[idxs[0]],y[idxs[0]])
        continue
      feature=self.__choose_feature(X_scale,idxs)
      median_idx=self.__get_median_idx(X,idxs,feature)
      idxs_left, idxs_right = self._split_feature(X, idxs, feature, median_idx)
      nd.feature = feature
      nd.split = (X[median_idx], y[median_idx])
      if idxs_left != []:
        nd.left = Node()
        nd.left.father = nd
        que.append((nd.left, idxs_left))
      if idxs_right != []:
        nd.right = Node()
        nd.right.father = nd
        que.append((nd.right, idxs_right))

  def _search(self,Xi,nd):
    while nd.left or nd.right:
      if nd.left is None:
        nd = nd.right
      elif nd.right is None:
        nd = nd.left
      else:
        if Xi[nd.feature] < nd.split[0][nd.feature]:
          nd = nd.left
        else:
          nd = nd.right
    return nd

  def _get_eu_dist(self, Xi, nd):
    X0 = nd.split[0]
    return get_euclidean_distance(Xi, X0)

  def _get_hyper_plane_dist(self, Xi, nd):
    j = nd.feature
    X0 = nd.split[0]
    return (Xi[j] - X0[j]) ** 2

  def nearest_neighbour_search(self, Xi):
    dist_best = float("inf")
    nd_best = self._search(Xi, self.root)
    que = [(self.root, nd_best)]
    while que:
      nd_root, nd_cur = que.pop(0)
      while 1:
        dist = self._get_eu_dist(Xi, nd_cur)
        if dist < dist_best:
          dist_best = dist
          nd_best = nd_cur
        if nd_cur is not nd_root:
          nd_bro = nd_cur.brother
          if nd_bro is not None:
            dist_hyper = self._get_hyper_plane_dist(
              Xi, nd_cur.father)
            if dist > dist_hyper:
              _nd_best = self._search(Xi, nd_bro)
              que.append((nd_bro, _nd_best))
          nd_cur = nd_cur.father
        else:
          break
    return nd_best

def main():
  low=0
  high=100
  n_rows=1000
  n_cols=2
  X = gen_data(low, high, n_rows, n_cols)
  y = gen_data(low, high, n_rows)
  Xi = gen_data(low, high, n_cols)
  tree = KDTree()

  tree.buildTree(X, y)

  nd = tree.nearest_neighbour_search(np.array(Xi))
  print(nd)

if __name__=='__main__':
  main()