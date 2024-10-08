import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = x.T

print(x.shape)
print(y.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(1000, input_dim=3))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4. 평가 예측
loss = model.evaluate(x,y)
results = model.predict([[10,1.3,0]])
print('로스 :', loss)
print('--의 예측값 :', results)


