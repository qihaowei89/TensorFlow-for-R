# install.packages("devtools")
devtools::install_github("rstudio/keras",force = TRUE)

library(keras)
library(magrittr)
install_keras(tensorflow = "gpu")
# ?install_keras
# mnist <- dataset_mnist()
# boston <- dataset_boston_housing()
# cifar <- dataset_cifar100()
# cifar10 <- dataset_cifar10()
# fashion_mnista <- dataset_fashion_mnist()
# imdb <- dataset_imdb()
# reuters <- dataset_reuters() # not yet 
# 
# save(mnist,file = "/home/wqh/work/TensorFlow-for-R/mnist.RData")
# save(boston,file = "/home/wqh/work/TensorFlow-for-R/boston.RData")
# save(cifar,file = "/home/wqh/work/TensorFlow-for-R/cifar100.RData")
# save(cifar10,file = "/home/wqh/work/TensorFlow-for-R/cifar10.RData")
# save(fashion_mnista,file = "/home/wqh/work/TensorFlow-for-R/fashion_mnista.RData")
# save(imdb ,file = "/home/wqh/work/TensorFlow-for-R/imdb.RData")
# save(reuters,file = "/home/wqh/work/TensorFlow-for-R/reuters.RData")
# 
minst <- load("dataset/mnist.RData")
x_train <- mnist$train$x
x_x <-mnist$train$x
dim(x_train)
y_train <- mnist$train$y
dim(y_train)
x_test <- mnist$test$x
dim(x_test)
y_test <- mnist$test$y
dim(y_test)

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
dim(x_train)
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
dim(x_test)
# rescale
x_train <- x_train / 255
x_test <- x_test / 255


y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')



model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

plot(history)
model %>% evaluate(x_test, y_test)
model %>% predict_classes(x_test)


