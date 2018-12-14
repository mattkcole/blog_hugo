# create model
model <- keras_model_sequential()

# define and compile the model
model %>% 
        layer_dense(units = 32, activation = 'relu', input_shape = c(172800)) %>% 
        layer_dense(units = 10, activation = 'softmax') %>% 
        compile(
                optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics = c('accuracy')
        )

# Generate dummy data
data <- matrix(runif(1000*100), nrow = 1000, ncol = 100)
labels <- matrix(round(runif(37, min = 0, max = 1)), nrow = 1000, ncol = 1)

# Convert labels to categorical one-hot encoding
one_hot_labels <- to_categorical(labels, num_classes = 2)

# Train the model, iterating on the data in batches of 32 samples
model %>% fit(x_train, one_hot_labels, epochs=10, batch_size=32)
