def resnet_archi():
    
    from keras.applications.resnet50 import ResNet50
    from keras.models import Model
    import keras
    restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
    output = restnet.layers[-1].output
    output = keras.layers.Flatten()(output)
    restnet = Model(restnet.input, output=output)
    for layer in restnet.layers:
        layer.trainable = False
    import pandas as pd
    restnet.trainable = True
    set_trainable = False
    for layer in restnet.layers:
        if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    layers = [(layer, layer.name, layer.trainable) for layer in restnet.layers]
    from keras.models import Sequential
    from keras import optimizers
    model_finetuned = Sequential()
    model_finetuned.add(restnet)
    model_finetuned.add(Dense(512, activation='relu', input_dim=(224,224,1)))
    model_finetuned.add(Dropout(0.5))
    model_finetuned.add(Dense(3, activation='sigmoid'))
    model_finetuned.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])
    return model_finetuned
