# Framework for IEEE course final project
# Fan Cheng, 2022

import random
random.seed()


class Matrix:

	def __init__(self, data=None, dim=None, init_value=0.0) -> None:
		"""
		构造函数，可以根据data或dim创建一个矩阵。当data和dim都为None时抛出异常。
		当data不为None时，自动推断dim。若dim不为None，则用init_value初始化矩阵。
		"""
		if data is None and dim is None:
			raise Exception("It's illegal that both data and dim are None!")

		if data is not None:
			self.data = [[float(element) for element in row] for row in data]

			self.dim = (len(data), len(data[0]) if len(data) > 0 else 0)
		else:
			try:
				if not isinstance(dim, (tuple, list)) or len(dim) != 2:
					raise TypeError("dim should be a tuple or list of two integers")
				data = [[init_value for _ in range(dim[1])] for _ in range(dim[0])]
				self.data = [[float(element) for element in row] for row in data]
				self.dim = (len(data), len(data[0]) if len(data) > 0 else 0)
			except TypeError as e:
				print("Here occurs a TypeError")
				raise e
				
	def shape(self):
		r"""
		返回矩阵的形状 dim
		"""
		return self.dim
		
	def reshape(self, newdim):
		"""
		将矩阵从(m,n)维拉伸为newdim=(m1,n1)
		该函数不改变 self
		
		Args:
			newdim: 一个元组 (m1, n1) 表示拉伸后的矩阵形状。如果 m1 * n1 不等于 self.dim[0] * self.dim[1],
					应抛出异常
		
		Returns:
			Matrix: 一个 Matrix 类型的返回结果, 表示 reshape 得到的结果
		"""
		# 克隆 self 的副本，避免更改 self 的任何元素
		clone = self.copy()
		# 检查新的形状是否与原始形状兼容
		if clone.dim[0] * clone.dim[1] != newdim[0] * newdim[1]:
			raise ValueError("The new shape must be compatible with the original shape.")
		# 将矩阵展平，然后重新排列
		flat = [elem for row in clone.data for elem in row]
		new_data = [flat[i * newdim[1]:(i + 1) * newdim[1]] for i in range(newdim[0])]
		return Matrix(data=new_data, dim=newdim)

	def dot(self, other):
		r"""
		矩阵乘法：矩阵乘以矩阵
		按照公式 A[i, j] = \sum_k B[i, k] * C[k, j] 计算 A = B.dot(C)

		Args:
			other: 参与运算的另一个 Matrix 实例
		
		Returns:
			Matrix: 计算结果
		
		Examples:
			>>> A = Matrix(data=[[1, 2], [3, 4]])
			>>> A.dot(A)
			>>> [[ 7 10]
				[15 22]]
		"""
		# 检查矩阵维度是否兼容
		if self.dim[1] != other.dim[0]:
			raise ValueError("The number of columns in the first matrix must match the number of rows in the second matrix.")

		# 计算矩阵乘法结果
		rows, cols = self.dim[0], other.dim[1]
		result = [[0 for _ in range(cols)] for _ in range(rows)]
		for i in range(rows):
			for j in range(cols):
				for k in range(self.dim[1]):
					result[i][j] += self.data[i][k] * other.data[k][j]
		return Matrix(data=result)

	def T(self):
		r"""
		矩阵的转置

		Returns:
			Matrix: 矩阵的转置

		Examples:
			>>> A = Matrix(data=[[1, 2], [3, 4]])
			>>> A.T()
			>>> [[1 3]
				 [2 4]]
			>>> B = Matrix(data=[[1, 2, 3], [4, 5, 6]])
			>>> B.T()
			>>> [[1 4]
				 [2 5]
				 [3 6]]
		"""
		# 克隆 self 的副本，避免更改 self 的任何元素
		clone = self.copy()
		# 转置矩阵
		transposed = [[clone.data[r][c] for r in range(clone.dim[0])] for c in range(clone.dim[1])]
		return Matrix(data=transposed)

	def sum(self, axis=None): 
		r"""
		根据指定的坐标轴对矩阵元素进行求和

		Args:
			axis: 一个整数，或者 None. 默认值: None
				  axis = 0 表示对矩阵进行按列求和，得到形状为 (1, self.dim[1]) 的矩阵
				  axis = 1 表示对矩阵进行按行求和，得到形状为 (self.dim[0], 1) 的矩阵
				  axis = None 表示对矩阵全部元素进行求和，得到形状为 (1, 1) 的矩阵
		
		Returns:
			Matrix: 一个 Matrix 类的实例，表示求和结果

		Examples:
			>>> A = Matrix(data=[[1, 2, 3], [4, 5, 6]])
			>>> A.sum()
			>>> [[21]]
			>>> A.sum(axis=0)
			>>> [[5 7 9]]
			>>> A.sum(axis=1)
			>>> [[6]
				 [15]]
		"""
		# 克隆 self 的副本，避免更改 self 的任何元素
		clone = self.copy()
		match axis:
			case None:
				# 所有元素求和：先按列求和，再对结果按行求和
				col_sum = clone.sum(axis=0)
				return col_sum.sum(axis=1)

			case 0:
				# 按列求和：用全1行向量左乘矩阵
				ones_row = Matrix(data=[[1] * clone.dim[0]])
				return ones_row.dot(clone)

			case 1:
				# 按行求和：矩阵右乘全1列向量
				ones_col = Matrix(data=[[1]] * clone.dim[1])
				return clone.dot(ones_col)

			case _:
				raise ValueError("Axis must be None, 0, or 1")

	def copy(self):
		r"""
		返回matrix的一个备份

		Returns:
			Matrix: 一个self的备份
		"""
		# 深拷贝
		new_data = [row[:] for row in self.data]
		return Matrix(data=new_data)

	def Kronecker_product(self, other):
		r"""
		计算两个矩阵的Kronecker积，具体定义可以搜索，https://baike.baidu.com/item/克罗内克积/6282573

		Args:
			other: 参与运算的另一个 Matrix

		Returns:
			Matrix: Kronecke product 的计算结果
		"""
		# 克隆 self 的副本，避免更改 self 的任何元素
		clone = self.copy()
		r, c = clone.dim

		# 先列后行
		result = []
		temp = []
		for i in range(clone.dim[0]):
			for j in range(clone.dim[1]):
				x = clone.data[i][j]
				temp.append(num_mul(other, x))
			result.append(concatenate(temp, axis=1))
			#print(concatenate(temp, axis=1))
			temp.clear()
		#print(concatenate(result, axis=0))
		# 将行拼接起来
		return concatenate(result, axis=0)
	
	def __getitem__(self, key):
		r"""
		实现 Matrix 的索引功能，即 Matrix 实例可以通过 [] 获取矩阵中的元素（或子矩阵）

		x[key] 具备以下基本特性：
		1. 单值索引
			x[a, b] 返回 Matrix 实例 x 的第 a 行, 第 b 列处的元素 (从 0 开始编号)
		2. 矩阵切片
			x[a:b, c:d] 返回 Matrix 实例 x 的一个由 第 a, a+1, ..., b-1 行, 第 c, c+1, ..., d-1 列元素构成的子矩阵
			特别地, 需要支持省略切片左(右)端点参数的写法, 如 x 是一个 n 行 m 列矩阵, 那么
			x[:b, c:] 的语义等价于 x[0:b, c:m]
			x[:, :] 的语义等价于 x[0:n, 0:m]
		Args:
			key: 一个元组，表示索引
		Returns:
			索引结果，单个元素或者矩阵切片
		"""

		# 如果传入的 key 不是元组
		if not isinstance(key, tuple):
			key = (key, )

		# 索引必须包含两个元素；否则抛出异常
		if len(key) != 2:
			raise IndexError("Matrix indexing requires two indices")

		row_idx, col_idx = key
		n, m = self.dim

		# 行索引处理
		if isinstance(row_idx, int):
			# 检查索引是否越界
			if row_idx < -n or row_idx >= n:
				raise IndexError(f"Row index {row_idx} out of range")
			# 处理负数索引
			row_idx = row_idx % n
			# 存储行索引
			row_indices = [row_idx]
		elif isinstance(row_idx, slice):
			start = row_idx.start if row_idx.start is not None else 0
			stop = row_idx.stop if row_idx.stop is not None else n
			step = row_idx.step if row_idx.step is not None else 1
			# 处理负索引
			if start < 0:
				start = n + start
			if stop < 0:
				stop = n + stop
			
			row_indices = list(range(start, stop, step))
		else:
			# 不支持其他类型索引
			raise TypeError(f"Row index must be int or slice, not {type(row_idx).__name__}")

		# --- 列索引处理 ---
		if isinstance(col_idx, int):
			# 如果是整数（单列）：支持负数索引
			if col_idx < -m or col_idx >= m:
				raise IndexError(f"Column index {col_idx} out of range")
			col_idx = col_idx % m
			col_indices = [col_idx]
		elif isinstance(col_idx, slice):
			start = col_idx.start if col_idx.start is not None else 0
			stop = col_idx.stop if col_idx.stop is not None else m
			step = col_idx.step if col_idx.step is not None else 1
			if start < 0:
				start = m + start
			if stop < 0:
				stop = m + stop
			col_indices = list(range(start, stop, step))
		else:
			raise TypeError(f"Column index must be int or slice, not {type(col_idx).__name__}")


		# 如果索引都是 int，直接返回元素值
		if isinstance(row_idx, int) and isinstance(col_idx, int):
			return self.data[row_indices[0]][col_indices[0]]

		# 否则返回一个新 Matrix 实例，数据是对应的子矩阵
		result_data = []
		for i in row_indices:
			# 依次提取每一指定行的指定列
			row = []
			for j in col_indices:
				row.append(self.data[i][j])
			result_data.append(row)

		return Matrix(data=result_data)

	def __setitem__(self, key, value):
		r"""
		实现 Matrix 的赋值功能, 通过 x[key] = value 进行赋值的功能

		类似于 __getitem__ , 需要具备以下基本特性:
		1. 单元素赋值
			x[a, b] = k 的含义为，将 Matrix 实例 x 的 第 a 行, 第 b 处的元素赋值为 k (从 0 开始编号)
		2. 对矩阵切片赋值
			x[a:b, c:d] = value 其中 value 是一个 (b-a)行(d-c)列的 Matrix 实例
			含义为, 将由 Matrix 实例 x 的第 a, a+1, ..., b-1 行, 第 c, c+1, ..., d-1 列元素构成的子矩阵 赋值为 value 矩阵
			即 子矩阵的 (i, j) 位置赋值为 value[i, j]
			同样地, 这里也需要支持如 x[:b, c:] = value, x[:, :] = value 等省略写法
		
		Args:
			key: 一个元组，表示索引
			value: 赋值运算的右值，即要赋的值

		Examples:
			>>> x = Matrix(data=[
						[0, 1, 2, 3],
						[4, 5, 6, 7],
						[8, 9, 0, 1]
					])
			>>> x[1, 2] = 0
			>>> x
			>>> [[0 1 2 3]
				 [4 5 0 7]
				 [8 9 0 1]]
			>>> x[1:, 2:] = Matrix(data=[[1, 2], [3, 4]])
			>>> x
			>>> [[0 1 2 3]
				 [4 5 1 2]
				 [8 9 3 4]]
		"""
		if not isinstance(key, tuple):
			key = (key,)
		if len(key) != 2:
			raise IndexError("Matrix indexing requires exactly two indices")

		row_idx, col_idx = key

		# 处理索引
		n, m = self.dim
				# 行索引处理
		if isinstance(row_idx, int):
			# 检查索引是否越界
			if row_idx < -n or row_idx >= n:
				raise IndexError(f"Row index {row_idx} out of range")
			# 处理负数索引
			row_idx = row_idx % n
			# 存储行索引
			row_indices = [row_idx]
		elif isinstance(row_idx, slice):
			start = row_idx.start if row_idx.start is not None else 0
			stop = row_idx.stop if row_idx.stop is not None else n
			step = row_idx.step if row_idx.step is not None else 1
			# 处理负索引
			if start < 0:
				start = n + start
			if stop < 0:
				stop = n + stop
			
			row_indices = list(range(start, stop, step))
		else:
			# 不支持其他类型索引
			raise TypeError(f"Row index must be int or slice, not {type(row_idx).__name__}")

		# 列索引处理
		if isinstance(col_idx, int):
			# 如果是整数（单列）：支持负数索引
			if col_idx < -m or col_idx >= m:
				raise IndexError(f"Column index {col_idx} out of range")
			col_idx = col_idx % m
			col_indices = [col_idx]
		elif isinstance(col_idx, slice):
			start = col_idx.start if col_idx.start is not None else 0
			stop = col_idx.stop if col_idx.stop is not None else m
			step = col_idx.step if col_idx.step is not None else 1
			if start < 0:
				start = m + start
			if stop < 0:
				stop = m + stop
			col_indices = list(range(start, stop, step))
		else:
			raise TypeError(f"Column index must be int or slice, not {type(col_idx).__name__}")

		if isinstance(row_idx, int) and isinstance(col_idx, int):
			if isinstance(value, float):
				self.data[row_idx][col_idx] = value
				return None
			elif not isinstance(value, float):
				raise TypeError(f"The value must be float, not {type(value).__name__}, if you want to replace a certain element with the value given.")

		if not isinstance(value, Matrix):
			raise TypeError("The value must be a Matrix if you want to replace a certain area of elements.")
		# ensure submatrix shape matches the assignment region
		if len(row_indices) != value.dim[0] or len(col_indices) != value.dim[1]:
			raise ValueError("Shape of value does not match the target slice.")
		
		#依次替换
		for i_target, i_value in enumerate(row_indices):
			for j_target, j_value in enumerate(col_indices):
				self.data[i_value][j_value] = value.data[i_target][j_target]
		return None

	def __pow__(self, n):
		r"""
		矩阵的n次幂，n为自然数
		该函数应当不改变 self 的内容

		Args:
			n: int, 自然数

		Returns:
			Matrix: 运算结果
		"""
		if not isinstance(n, int) or n < 0:
			raise ValueError("n must be a non-negative integer.")
		if self.dim[1] != self.dim[0]:
			raise Exception("Matrix must be square for power operation.")
		if n == 0:
			# 返回单位矩阵
			return I(self.dim[0])
		result = self.copy()
		for _ in range(n - 1):
			result = result.dot(self)
		return result


	def __add__(self, other):
		r"""
		两个矩阵相加
		该函数应当不改变 self 和 other 的内容

		Args:
			other: 一个 Matrix 实例
		
		Returns:
			Matrix: 运算结果
		"""
		if self.dim != other.dim:
			raise Exception("The shapes of both matrix should be coresponded in order to add.")

		r, c = self.dim
		result = [[None for _ in range(c)] for _ in range(r)]
		for i in range(r):
			for j in range(c):
				result[i][j] = self.data[i][j] + other.data[i][j]

		return Matrix(data=result)

	def __sub__(self, other):
		r"""
		两个矩阵相减
		该函数应当不改变 self 和 other 的内容

		Args:
			other: 一个 Matrix 实例
		
		Returns:
			Matrix: 运算结果
		"""
		if self.dim != other.dim:
			raise Exception("The shapes of both matrix should be coresponded in order to sub.")

		r, c = self.dim
		result = [[None for _ in range(c)] for _ in range(r)]
		for i in range(r):
			for j in range(c):
				result[i][j] = self.data[i][j] - other.data[i][j]

		return Matrix(data=result)

	def __mul__(self, other):
		r"""
		两个矩阵 对应位置 元素  相乘
		注意 不是矩阵乘法dot
		该函数应当不改变 self 和 other 的内容

		Args:
			other: 一个 Matrix 实例
		
		Returns:
			Matrix: 运算结果

		Examples:
			>>> Matrix(data=[[1, 2]]) * Matrix(data=[[3, 4]])
			>>> [[3 8]]
		"""
		if self.dim != other.dim:
			raise Exception("The shapes of both matrix should be coresponded in order to multiply.")

		r, c = self.dim
		result = [[None] * c] * r
		for i in range(r):
			for j in range(c):
				result[i][j] = self.data[i][j] * other.data[i][j]

		return Matrix(data=result)


	def __len__(self):
		r"""
		返回矩阵元素的数目

		Returns:
			int: 元素数目，即 行数 * 列数
		"""
		return self.dim[1] * self.dim[0]

	def __str__(self):
		r"""
		按照
		[[  0   1   4   9  16  25  36  49]
	 	[ 64  81 100 121 144 169 196 225]
	 	[256 289 324 361 400 441 484 529]]
	 	的格式将矩阵表示为一个 字符串
	 	！！！ 注意返回值是字符串
		"""
		# 找到每列的最大宽度
		max_widths = []
		for j in range(self.dim[1]):
			max_width = 0
			for i in range(self.dim[0]):
				width = len(str(self.data[i][j]))
				if width > max_width:
					max_width = width
			max_widths.append(max_width)

		# 构建格式化字符串
		lines = []
		for i in range(self.dim[0]):
			elements = []
			for j in range(self.dim[1]):
				# 右对齐，宽度为该列最大值
				elements.append(str(self.data[i][j]).rjust(max_widths[j]))
			lines.append("[" + " ".join(elements) + "]")

		return "[" + ",\n ".join(lines) + "]"

	def det(self):
		r"""
		计算方阵的行列式。对于非方阵的情形应抛出异常。
		要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
		提示: Gauss消元
		
		Returns:
			一个 Python int 或者 float, 表示计算结果
		"""
		# 已经在gauss_elimination实现
		return self.gauss_elimination(det= True)

	def inverse(self):
		r"""
		计算非奇异方阵的逆矩阵。对于非方阵或奇异阵的情形应抛出异常。
		要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
		提示: Gauss消元

		Returns:
			Matrix: 一个 Matrix 实例，表示逆矩阵
		"""
		clone = self.copy()
		rows, cols = clone.dim
		if rows != cols:
			return None, "所给矩阵必须是方阵"
		if self.det() == 0.0:
			return None, "矩阵不可逆，行列式为0"

		n = clone.dim[0]
		augmented = []
		for i in range(n):
			row = clone.data[i][:]  # 原矩阵部分
			# 单位矩阵部分
			row.extend([1.0 if j == i else 0.0 for j in range(n)])
			augmented.append(row)

		for col in range(n):
			# 寻找主元行
			pivot_row = -1
			for r in range(col, n):
				if abs(augmented[r][col]) > 1e-10:
					pivot_row = r
					break

			if pivot_row == -1:
				return None, "矩阵不可逆"

			# 交换行（如果需要）
			if pivot_row != col:
				augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

			# 归一化主元行
			pivot_val = augmented[col][col]
			for j in range(2 * n):
				augmented[col][j] /= pivot_val

			# 消去其他行
			for r in range(n):
				if r != col:
					factor = augmented[r][col]
					for j in range(2 * n):
						augmented[r][j] -= augmented[col][j] * factor

		# 提取逆矩阵
		inverse = []
		for i in range(n):
			inverse.append(augmented[i][n:])

		return Matrix(data=inverse)

	def rank(self):
		clone = self.copy()
		clone_gauss = clone.gauss_elimination()
		r, c = clone.dim
		m = max(r, c) + 1
		for i in range(m):
			if clone_gauss.data[i][i] == 0:
				return i

		return m

	def gauss_elimination(self, det=False):
		"""
		Gauss elimination for row reduction or determinant calculation.

		Args:
			det (bool): 
				- False: return row-echelon form as a Matrix.
				- True: return determinant as float (input must be square matrix).

		Returns: 
			Matrix (if det=False) or float (if det=True).
		"""
		clone = self.copy()
		temp_matrix = [row[:] for row in clone.data]  # 创建复制，防止修改原始对象
		rows, cols = clone.dim

		if not det:
			for i in range(min(rows, cols)):
				# 找出最大主元行
				pivot_row = max(range(i, rows), key=lambda r: abs(temp_matrix[r][i]))
				if abs(temp_matrix[pivot_row][i]) < 1e-12:
					continue
				if pivot_row != i:
					temp_matrix[i], temp_matrix[pivot_row] = temp_matrix[pivot_row], temp_matrix[i]

				pivot = temp_matrix[i][i]
				# 标准化行首非零元
				for j in range(i, cols):
					temp_matrix[i][j] /= pivot

				# Eliminate below
				for r in range(i + 1, rows):
					factor = temp_matrix[r][i]
					for j in range(i, cols):
						temp_matrix[r][j] -= temp_matrix[i][j] * factor

			return Matrix(data=temp_matrix)
		else:
			if rows != cols:
				raise ValueError("Square matrix required for determinant.")
			n = rows
			sign = 1
			det_val = 1.0
			for i in range(n):
				# 找出最大主元行
				pivot_row = max(range(i, n), key=lambda r: abs(temp_matrix[r][i]))
				if abs(temp_matrix[pivot_row][i]) < 1e-12:
					return 0.0
				if pivot_row != i:
					temp_matrix[i], temp_matrix[pivot_row] = temp_matrix[pivot_row], temp_matrix[i]
					sign *= -1
				pivot = temp_matrix[i][i]
				det_val *= pivot
				# 删除下行
				for r in range(i+1, n):
					factor = temp_matrix[r][i] / pivot
					for j in range(i, n):
						temp_matrix[r][j] -= temp_matrix[i][j] * factor
			return sign * det_val

	def __eq__(self, other):
		if self.shape() != other.shape():
			return False
		for x_r, y_r in zip(self.data, other.data):
			for x_i, y_i in zip(x_r, y_r):
				if abs(x_i - y_i) < 1e-10:
					pass
				else:
					return False
		return True
	
	def Jordan_normal_form(self):
		"""
		计算矩阵的 Jordan 标准形，要求该函数不改变 self 的内容，时间复杂度不超过 O(n**3)，提示：特征值分解
		Returns:
			Matrix: 一个 Matrix 实例，表示 Jordan 标准形
		"""
		# 只处理方阵
		if self.dim[0] != self.dim[1]:
			raise ValueError("Matrix must be square to compute Jordan normal form.")

		n = self.dim[0]
		if n == 0:
			return Matrix(data=[])

		# 将原矩阵数据转换为内部计算类型
		A = [[float(self.data[i][j]) for j in range(n)] for i in range(n)]
		tol = 1e-8

		def eye(size):
			return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]

		def mat_add(X, Y):
			return [[X[i][j] + Y[i][j] for j in range(n)] for i in range(n)]

		def mat_scalar_mul(X, scalar):
			return [[X[i][j] * scalar for j in range(n)] for i in range(n)]

		def mat_mul(X, Y):
			result = [[0.0] * n for _ in range(n)]
			for i in range(n):
				for j in range(n):
					sum_val = 0.0
					for k in range(n):
						sum_val += X[i][k] * Y[k][j]
					result[i][j] = sum_val
			return result

		def mat_pow(X, power):
			result = eye(n)
			for _ in range(power):
				result = mat_mul(result, X)
			return result

		def trace(X):
			return sum(X[i][i] for i in range(n))

		def poly_eval(coeffs, x):
			value = 0+0j
			for coeff in coeffs:
				value = value * x + coeff
			return value

		def poly_derivative(coeffs):
			deg = len(coeffs) - 1
			return [coeffs[i] * (deg - i) for i in range(len(coeffs) - 1)]

		def find_root(coeffs):
			for attempt in range(50):
				guess = complex(random.random(), random.random())
				z = guess
				for _ in range(300):
					p = poly_eval(coeffs, z)
					dp = poly_eval(poly_derivative(coeffs), z)
					if abs(dp) < tol:
						break
					z_new = z - p / dp
					if abs(z_new - z) < tol:
						z = z_new
						break
					z = z_new
				if abs(poly_eval(coeffs, z)) < 1e-6:
					return z
			raise ValueError("Unable to find polynomial root for Jordan normal form.")

		def deflate(coeffs, root):
			new = [coeffs[0]]
			for coeff in coeffs[1:]:
				new.append(new[-1] * root + coeff)
			remainder = new.pop()
			return new, remainder

		def characteristic_polynomial():
			B = eye(n)
			coeffs = [1.0]
			for k in range(1, n + 1):
				AB = mat_mul(A, B)
				c = -trace(AB) / k
				coeffs.append(c)
				B = mat_add(AB, mat_scalar_mul(eye(n), c))
			return coeffs

		def row_rank(matrix):
			rows = len(matrix)
			cols = len(matrix[0]) if rows else 0
			M = [row[:] for row in matrix]
			r = 0
			for c in range(cols):
				if r >= rows:
					break
				pivot = max(range(r, rows), key=lambda i: abs(M[i][c]))
				if abs(M[pivot][c]) < tol:
					continue
				M[r], M[pivot] = M[pivot], M[r]
				factor = M[r][c]
				for j in range(c, cols):
					M[r][j] /= factor
				for i in range(rows):
					if i != r:
						factor = M[i][c]
						if abs(factor) < tol:
							continue
						for j in range(c, cols):
							M[i][j] -= factor * M[r][j]
				r += 1
			return r

		def nullity_for_eigenvalue(lam, power):
			M = [[A[i][j] - (lam if i == j else 0.0) for j in range(n)] for i in range(n)]
			P = eye(n)
			for _ in range(power):
				P = mat_mul(P, M)
			return n - row_rank(P)

		coeffs = characteristic_polynomial()
		poly = [complex(c) for c in coeffs]
		roots = []
		current = poly[:]
		for _ in range(n):
			root = find_root(current)
			if abs(root.imag) < 1e-8:
				root = complex(root.real, 0.0)
			roots.append(root)
			current, remainder = deflate(current, root)
			if abs(remainder) > 1e-4:
				# if deflation fails, add a small perturbation and continue
				root = complex(root.real + 1e-6, root.imag + 1e-6)
				current, remainder = deflate(current, root)

		# 聚类特征值
		distinct_eigenvalues = []
		for root in roots:
			assigned = False
			for group in distinct_eigenvalues:
				if abs(root - group[0]) < 1e-5:
					group.append(root)
					assigned = True
					break
			if not assigned:
				distinct_eigenvalues.append([root])

		blocks = []
		for group in distinct_eigenvalues:
			lam = sum(group) / len(group)
			if abs(lam.imag) > 1e-6:
				raise ValueError("Complex eigenvalues are not supported by this implementation.")
			lam = float(lam.real)
			multiplicity = len(group)
			nullities = [0]
			for power in range(1, multiplicity + 1):
				nullities.append(nullity_for_eigenvalue(lam, power))
			m_values = []
			for power in range(1, multiplicity + 1):
				count_at_least = int(round(nullities[power] - nullities[power - 1]))
				m_values.append(max(count_at_least, 0))
			for k in range(1, multiplicity + 1):
				exact_count = m_values[k - 1] - (m_values[k] if k < multiplicity else 0)
				exact_count = max(int(round(exact_count)), 0)
				for _ in range(exact_count):
					block = [[0.0] * k for _ in range(k)]
					for i in range(k):
						block[i][i] = lam
						if i < k - 1:
							block[i][i + 1] = 1.0
					blocks.append(block)

		# 构建 Jordan 标准形
		total_size = sum(len(block) for block in blocks)
		J = [[0.0] * total_size for _ in range(total_size)]
		position = 0
		for block in blocks:
			size = len(block)
			for i in range(size):
				for j in range(size):
					J[position + i][position + j] = block[i][j]
			position += size

		return Matrix(data=J)
	
	def LU_decomposition(self):
		"""
		计算矩阵的 LU 分解，要求该函数不改变 self 的内容，时间复杂度不超过 O(n**3)
		Returns:
			L: 一个 Matrix 实例，表示下三角矩阵 L
			U: 一个 Matrix 实例，表示上三角矩阵 U
		"""
		n, m = self.dim
		if n != m:
			raise ValueError("LU decomposition requires a square matrix.")
		
		L = [[0.0] * n for _ in range(n)]
		U = [[0.0] * n for _ in range(n)]
		
		for i in range(n):
			L[i][i] = 1.0
		
		for j in range(n):
			for i in range(j + 1):
				U[i][j] = self.data[i][j]
				for k in range(i):
					U[i][j] -= L[i][k] * U[k][j]
			
			for i in range(j + 1, n):
				L[i][j] = self.data[i][j]
				for k in range(j):
					L[i][j] -= L[i][k] * U[k][j]
				if abs(U[j][j]) < 1e-12:
					raise ValueError("Matrix is singular and cannot be decomposed.")
				L[i][j] /= U[j][j]

		return Matrix(data=L), Matrix(data=U)
	
	def Cholesky_decomposition(self):
		"""
		计算矩阵的 Cholesky 分解，该函数不改变 self 的内容，时间复杂度不超过 O(n**3)
		Returns:
			L: 一个 Matrix 实例，表示下三角矩阵 L，使得 A = L * L^T
		"""
		n, m = self.dim
		if n != m:
			raise ValueError("Cholesky decomposition requires a square matrix.")
		
		L = [[0.0] * n for _ in range(n)]
		
		for i in range(n):
			for j in range(i + 1):
				sum_val = sum(L[i][k] * L[j][k] for k in range(j))
				if i == j:
					val = self.data[i][i] - sum_val
					if val <= 0:
						raise ValueError("Matrix is not positive definite.")
					L[i][j] = val ** 0.5
				else:
					if abs(L[j][j]) < 1e-12:
						raise ValueError("Matrix is not positive definite.")
					L[i][j] = (self.data[i][j] - sum_val) / L[j][j]

		return Matrix(data=L)
	
	def QR_decomposition(self):
		n, m = self.dim
		# 初始化 Q (n x m) 和 R (m x m) -> 窄分解
		Q = [[0.0] * m for _ in range(n)]
		R = [[0.0] * m for _ in range(m)]

		for j in range(m):
			# 提取当前列
			v = [self.data[i][j] for i in range(n)]
        
			for k in range(j):
				R[k][j] = sum(Q[i][k] * v[i] for i in range(n))
				for i in range(n):
					v[i] -= R[k][j] * Q[i][k]
        
        	# 计算模长
			norm_v = sum(x**2 for x in v) ** 0.5
			R[j][j] = norm_v
        
			if norm_v < 1e-12:
        	    # 说明列线性相关，如果是求逆或解方程则会失败
			    raise ValueError("Matrix has linearly dependent columns.")
			
			for i in range(n):
				Q[i][j] = v[i] / norm_v

		return Matrix(data=Q), Matrix(data=R)
	
	def eigenvalues(self):
		"""
		计算矩阵的特征值，要求该函数不改变 self 的内容，时间复杂度不超过 O(n**3)，提示：特征值分解
		Returns:
			List[float]: 一个列表，包含所有特征值
		"""
		J = self.Jordan_normal_form()
		eigenvalues = []
		for i in range(J.dim[0]):
			eigenvalues.append(J.data[i][i])
		return eigenvalues
	
	def SVD_decomposition(self):
		"""
		计算矩阵的 SVD 分解，要求该函数不改变 self 的内容，时间复杂度不超过 O(n**3)
		Returns:
			U: 一个 Matrix 实例，表示左奇异矩阵 U
			S: 一个 Matrix 实例，表示奇异值矩阵 S (对角矩阵)
			V: 一个 Matrix 实例，表示右奇异矩阵 V^T
		"""
		n, m = self.dim
		A = [[float(self.data[i][j]) for j in range(m)] for i in range(n)]
		
		# 计算 A^T * A 和 A * A^T
		A_T = [[A[i][j] for i in range(n)] for j in range(m)]
		A_T_A = [[sum(A_T[i][k] * A[k][j] for k in range(n)) for j in range(m)] for i in range(m)]
		A_A_T = [[sum(A[i][k] * A_T[k][j] for k in range(m)) for j in range(n)] for i in range(n)]

		# 计算 A^T * A 的特征值和特征向量 -> V 和 S^2
		V, S_squared = self._eigen_decomposition(A_T_A)
		
		# 计算 A * A^T 的特征值和特征向量 -> U 和 S^2
		U, _ = self._eigen_decomposition(A_A_T)

		S = [[0.0] * m for _ in range(n)]
		for i in range(min(n, m)):
			S[i][i] = S_squared[i] ** 0.5

		return Matrix(data=U), Matrix(data=S), Matrix(data=V)
	
	def _eigen_decomposition(self, matrix):
		# 这里我们可以使用幂迭代法来近似计算特征值和特征向量
		n = len(matrix)
		eigenvalues = []
		eigenvectors = []
		A = [row[:] for row in matrix]

		for _ in range(n):
			b_k = [random.random() for _ in range(n)]
			for _ in range(1000):
				b_k1 = [sum(A[i][j] * b_k[j] for j in range(n)) for i in range(n)]
				norm_b_k1 = sum(x**2 for x in b_k1) ** 0.5
				if norm_b_k1 < 1e-12:
					break
				b_k = [x / norm_b_k1 for x in b_k1]

			eigenvalue = sum(A[i][j] * b_k[j] for i in range(n) for j in range(n)) / sum(b_k[i] * b_k[i] for i in range(n))
			eigenvalues.append(eigenvalue)
			eigenvectors.append(b_k)

			# Deflation
			for i in range(n):
				for j in range(n):
					A[i][j] -= eigenvalue * b_k[i] * b_k[j]

		return eigenvectors, eigenvalues



def I(n):
	"""
	返回一个 n*n 单位矩阵
	"""
	if not isinstance(n, int) or n <= 0:
		raise ValueError("I(n): n should be a positive integer.")
	data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
	return Matrix(data=data, dim=(n, n))

def narray(dim, init_value=1): # dim (,,,,,), init为矩阵元素初始值
	r"""
	返回一个matrix，维数为dim，初始值为init_value
	
	Args:
		dim: Tuple[int, int] 表示矩阵形状
		init_value: 表示初始值，默认值: 1

	Returns:
		Matrix: 一个 Matrix 类型的实例
	"""
	return Matrix(dim=dim, init_value=init_value)

def arange(start, end, step):
	r"""
	返回一个1*n 的 narray 其中的元素类同 range(start, end, step)

	Args:
		start: 起始点(包含)
		end: 终止点(不包含)
		step: 步长

	Returns:
		Matrix: 一个 Matrix 实例
	"""
	# 生成1*n的矩阵，每个元素都是start到end-1之间的整数，步长为step
	result = [[float(x) for x in range(start, end, step)]]
	return Matrix(data=result)

def zeros(dim: tuple):
	r"""
	返回一个维数为dim 的全0 narray

	Args:
		dim: Tuple[int, int] 表示矩阵形状

	Returns:
		Matrix: 一个 Matrix 类型的实例
	"""
	x = dim
	# 生成全0矩阵
	result = [[0.0 for _ in range(dim[1])] for _ in range(dim[0])]
	return Matrix(data=result)

def zeros_like(matrix):
	r"""
	返回一个形状和matrix一样 的全0 narray

	Args:
		matrix: 一个 Matrix 实例
	
	Returns:
		Matrix: 一个 Matrix 类型的实例

	Examples:
		>>> A = Matrix(data=[[1, 2, 3], [2, 3, 4]])
		>>> zeros_like(A)
		>>> [[0 0 0]
			 [0 0 0]]
	"""
	# 生成相似于Matrix的0矩阵
	result = [[0.0 for _ in range(matrix.dim[1])] for _ in range(matrix.dim[0])]
	return Matrix(data=result)

def ones(dim):
	r"""
	返回一个维数为dim 的全1 narray
	类同 zeros
	"""
	return Matrix(dim=dim, init_value=1.0)

def ones_like(matrix):
	r"""
	返回一个维数和matrix一样 的全1 narray
	类同 zeros_like
	"""
	clone = matrix.copy()
	return Matrix(dim=clone.dim, init_value=1.0)

def nrandom(dim):
	r"""
	返回一个维数为dim 的随机 narray
	参数与返回值类型同 zeros
	"""
	# 生成随机矩阵，每个元素都是1到100之间的随机整数
	result = [[random.randint(1, 100) for _ in range(dim[1])] for _ in range(dim[0])]
	return Matrix(data=result)

def nrandom_like(matrix: Matrix):
	r"""
	返回一个维数和matrix一样 的随机 narray
	参数与返回值类型同 zeros_like
	"""
	# 生成相似于Matrix的随机矩阵，每个元素都是1到100之间的随机整数
	result = [[random.randint(1, 100) for _ in range(matrix.dim[1])] for _ in range(matrix.dim[0])]
	return Matrix(data=result)

def concatenate(items, axis=0):
	r"""
	将若干矩阵按照指定的方向拼接起来
	若给定的输入在形状上不对应，应抛出异常
	该函数应当不改变 items 中的元素

	Args:
		items: 一个可迭代的对象，其中的元素为 Matrix 类型。
		axis: 一个取值为 0 或 1 的整数，表示拼接方向，默认值 0.
			  0 表示在第0维即行上进行拼接
			  1 表示在第1维即列上进行拼接
	
	Returns:
		Matrix: 一个 Matrix 类型的拼接结果

	Examples:
		>>> A, B = Matrix([[0, 1, 2]]), Matrix([[3, 4, 5]])
		>>> concatenate((A, B))
		>>> [[0 1 2]
			 [3 4 5]]
		>>> concatenate((A, B, A), axis=1)
		>>> [[0 1 2 3 4 5 0 1 2]]
	"""
	# 拷贝第一个矩阵
	temp = items[0].copy()
	temp_matrix = temp.data
	rows , cols = temp.dim
	if axis == 0: # 按行拼接
		for x in range(len(items) - 1):
			if items[x].dim[1] != items[x + 1].dim[1]: # 检查是否在形状上对应
				raise Exception("The shapes of the matrices are not corresponding.") # 抛出异常
		for x in range(1, len(items)): # 拼接其他矩阵
			temp_matrix += items[x].data 
		return Matrix(data=temp_matrix)
	elif axis == 1:
		for x in range(len(items) - 1): # 检查是否在形状上对应
			if items[x].dim[0] != items[x + 1].dim[0]:
				raise Exception("The shapes of the matrices are not corresponding.") # 抛出异常
		for r in range(rows):
			for x in range(1, len(items)): # 拼接其他矩阵
				temp_matrix[r].extend(items[x].data[r]) 
		return Matrix(data=temp_matrix)

def vectorize(func):
	r"""
	将给定函数进行向量化
	
	Args:
		func: 一个Python函数
	
	Returns:
		一个向量化的函数 F: Matrix -> Matrix, 它的参数是一个 Matrix 实例 x, 返回值也是一个 Matrix 实例；
		它将函数 func 作用在 参数 x 的每一个元素上
	
	Examples:
		>>> def func(x):
				return x ** 2
		>>> F = vectorize(func)
		>>> x = Matrix([[1, 2, 3],[2, 3, 1]])
		>>> F(x)
		>>> [[1 4 9]
			 [4 9 1]]
		>>> 
		>>> @vectorize
		>>> def my_abs(x):
				if x < 0:
					return -x
				else:
					return x
		>>> y = Matrix([[-1, 1], [2, -2]])
		>>> my_abs(y)
		>>> [[1, 1]
			 [2, 2]]
	"""
	# 定义一个向量化的函数
	def vectorized_func(M: Matrix) -> Matrix:
		temp = M.data
		r, c = M.dim
		result = [[func(temp[i][j]) for j in range(c)] for i in range(r)] # 将函数作用在每个元素上
		return Matrix(data=result)
	# 返回向量化的函数
	return vectorized_func

def num_mul(m : Matrix, n):
		"""
		实现矩阵的数乘操作，返回数乘后的新矩阵，原矩阵不变。

		Args:
			n (float or int): 用于数乘的数值

		Returns:
			Matrix: 数乘结果的新矩阵
		"""
		# 使用向量化方式实现高效数乘，避免修改自身数据
		return vectorize(lambda x: n * x)(m)

# 测试
if __name__ == "__main__":
	print("test here")
	C = Matrix(data=[[2,-1],[2,2]])
	print("C:")
	print(C)
	print("C SVD:")
	U, S, V = C.SVD_decomposition()
	print("U:")
	print(U)
	print("S:")
	print(S)
	print("V^T:")
	print(V)