library(tensorflow)
#install_tensorflow()

library(reticulate)
library(keras)
library(dplyr)
library(stringr)
library(caret)
library(imager)

use_python("~/anaconda3/envs/tf_image/bin/python")

train_gen <- image_data_generator(rescale = 1/255,
                                  horizontal_flip = T,
                                  vertical_flip = T,
                                  rotation_range = 45,
                                  zoom_range = 0.25,
                                  validation_split = 0.2)


target_size= c(64,64)
batch_size= 32

train_image <- flow_images_from_directory(directory = "/Users/david/Desktop/davidtuto1",
                                          target_size = target_size,
                                          color_mode="rgb",
                                          batch_size = batch_size,
                                          seed=123,
                                          subset="training",
                                          generator = train_gen
                                          )

val_image <- flow_images_from_directory(directory = "/Users/david/Desktop/davidtuto1",
                                          target_size = target_size,
                                          color_mode="rgb",
                                          batch_size = batch_size,
                                          seed=123,
                                          subset="validation",
                                          generator = train_gen
)


train_samples <-train_image$n

valid_samples <- val_image$n

output_n <- n_distinct(train_image$classes)



tensorflow::tf$random$set_seed(123)

model <- keras_model_sequential(name = "simple_model") %>% 
  
  # Convolution Layer
  layer_conv_2d(filters = 16,
                kernel_size = c(3,3),
                padding = "same",
                activation = "relu",
                input_shape = c(target_size, 3) 
  ) %>% 
  
  # Max Pooling Layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Flattening Layer
  layer_flatten() %>% 
  
  # Dense Layer
  layer_dense(units = 16,
              activation = "relu") %>% 
  
  # Output Layer
  layer_dense(units = output_n,
              activation = "softmax",
              name = "Output")



model %>% 
  keras::compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate =  0.01),
    metrics = "accuracy"
  )

# Fit data into model
history <- model %>% 
  fit(
    # training data
    train_image,
    
    # training epochs
    steps_per_epoch = as.integer(train_samples / batch_size), 
    epochs = 30, 
    
    # validation data
    validation_data = val_image,
    validation_steps = as.integer(valid_samples / batch_size)
  )

plot(history)


###################
val_data <- data.frame(file_name = paste0("/Users/david/Desktop/davidtuto1/", val_image$filenames)) %>% 
  mutate(class = str_extract(file_name, "pikachu|rondoudou|Carapuce|Salameche"))

head(val_data, 10)


#### Test
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = target_size, 
                      grayscale = F # Set FALSE if image is RGB
    )
    
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- x/255 # rescale image pixel
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

test_x <- image_prep(val_data$file_name)

# Check dimension of testing data set
dim(test_x)


pred_test <- model %>% predict(test_x) %>% k_argmax()

head(pred_test, 20)


### Permets de décoder
decode <- function(x){
  case_when(x == 2 ~ "pikachu",
            x == 1 ~ "Salameche",
            x == 0 ~ "Carapuce",
            x ==  3 ~ "rondoudou"
  )
}

pred_test <- sapply(as.array(pred_test), decode) 

head(pred_test, 20)
confusionMatrix(as.factor(pred_test), 
                as.factor(val_data$class)
)



##### Biggest model
tensorflow::tf$random$set_seed(123)

model_big <- keras_model_sequential() %>% 
  
  # First convolutional layer
  layer_conv_2d(filters = 32,
                kernel_size = c(5,5), # 5 x 5 filters
                padding = "same",
                activation = "relu",
                input_shape = c(target_size, 3)
  ) %>% 
  
  # Second convolutional layer
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3), # 3 x 3 filters
                padding = "same",
                activation = "relu"
  ) %>% 
  
  # Max pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Third convolutional layer
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                padding = "same",
                activation = "relu"
  ) %>% 
  
  # Max pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Fourth convolutional layer
  layer_conv_2d(filters = 128,
                kernel_size = c(3,3),
                padding = "same",
                activation = "relu"
  ) %>% 
  
  # Max pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Fifth convolutional layer
  layer_conv_2d(filters = 256,
                kernel_size = c(3,3),
                padding = "same",
                activation = "relu"
  ) %>% 
  
  # Max pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Flattening layer
  layer_flatten() %>% 
  
  # Dense layer
  layer_dense(units = 64,
              activation = "relu") %>% 
  
  # Output layer
  layer_dense(name = "Output",
              units = 3, 
              activation = "softmax")

model_big


model_big %>% 
  keras::compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adagrad(learning_rate = 0.001),
    metrics = "accuracy"
  )

history2 <- model %>% 
  keras::fit_generator(
    # training data
    train_image,
    
    # epochs
    steps_per_epoch = as.integer(train_samples / batch_size), 
    epochs = 50, 
    
    # validation data
    validation_data = val_image,
    validation_steps = as.integer(valid_samples / batch_size),
    
    # print progress but don't create graphic
    verbose = 1,
    view_metrics = 0
  )

plot(history2)

confusionMatrix(as.factor(pred_test), 
                as.factor(val_data$class)
)
