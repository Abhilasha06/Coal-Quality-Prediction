def cnn_archi():
  model=Sequential()
  model.add(InputLayer(input_shape=[64,64,1]))

  model.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=5,padding='same'))
          
  model.add(Conv2D(filters=32,kernel_size=5,strides=2,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=5,padding='same'))
          
  model.add(Conv2D(filters=64,kernel_size=3,strides=2,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=8,padding='same'))

  model.add(Conv2D(filters=128,kernel_size=3,strides=1,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=5,padding='same'))
          

  model.add(Flatten())


  model.add(Dense(2048,activation='relu'))
  model.add(Dropout(0.50))
  model.add(Dense(1024,activation='relu'))
  model.add(Dropout(0.50))
  model.add(Dense(512,activation='relu'))
  model.add(Dropout(0.50))
  model.add(Dense(3,activation='softmax'))
  optimizer=Adam(lr=1e-3)
          
  model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
  return model
