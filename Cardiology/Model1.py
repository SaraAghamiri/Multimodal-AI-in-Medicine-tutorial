input_clinical = Input(shape=(X_clinical_train.shape[1],))
x_clinical = Dense(64, activation='relu')(input_clinical)
x_clinical = Dense(32, activation='relu')(x_clinical)

input_image = Input(shape=(128, 128, 3))
x_image = Conv2D(32, (3, 3), activation='relu')(input_image)
x_image = MaxPooling2D((2, 2))(x_image)
x_image = Conv2D(64, (3, 3), activation='relu')(x_image)
x_image = MaxPooling2D((2, 2))(x_image)
x_image = Flatten()(x_image)
x_image = Dense(64, activation='relu')(x_image)

combined = concatenate([x_clinical, x_image])
output = Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[input_clinical, input_image], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit([X_clinical_train, X_image_train], y_train, 
                    epochs=20, batch_size=32, validation_split=0.2)
