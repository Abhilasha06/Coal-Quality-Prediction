def siamese_archi():

    left_input = Input((64,64,1))
    right_input = Input((64,64,1))

    # Base network architecture

    convnet=Sequential()
    convnet.add(InputLayer(input_shape=[64,64,1]))
    convnet.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu'))
    convnet.add(MaxPool2D(pool_size=5,padding='same'))   
    convnet.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu'))
    convnet.add(MaxPool2D(pool_size=5,padding='same'))  
    convnet.add(Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu'))
    convnet.add(MaxPool2D(pool_size=8,padding='same'))
    convnet.add(Conv2D(filters=128,kernel_size=3,strides=1,padding='same',activation='relu'))
    convnet.add(MaxPool2D(pool_size=5,padding='same'))   
    convnet.add(Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu'))
    convnet.add(Flatten())

    convnet.add(Dense(1024,activation='relu'))
    convnet.add(Dropout(0.20))
    convnet.add(Dense(512,activation='relu'))
    convnet.add(Dropout(0.25))
    convnet.add(Dense(64,activation='sigmoid'))

    #generating the encodings for two input images
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # Getting the L1 Distance between the 2 encodings
    L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

    # Add the distance function to the network
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam(0.0001, decay=2.5e-4)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
    return siamese_net
