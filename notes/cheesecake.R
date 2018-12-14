
library(dplyr)
library(jpeg)
library(keras)
library("EBImage")
library(devtools)
library(pryr)

cheesecakes = c('original cheesecake',
                'fresh strawberry cheesecake',
                "Reeses PB Choc Cake Cheesecake",
                "celebration cheesecake",
                "ChocolateHazelnutCrunch",
                "salted caramel cheesecake", #SaltedCaramelCheesecake
                "salted caramel cheesecake",
                "CoffeeCreamChocolateSupreme",
                "oreo dream extreme cheesecake",
                "ToastedMarsmallowSmoresGalore",
                "LemonMeringue",
                "adams peanut butter cup fudge ripple",
                "GodivaCheesecake",
                "RedVelvetCheesecake",
                "DulceDeLecheCheesecake",
                "WhiteChocolateRaspberryTruffle",
                "Social_ChrisOutrageousCheesecake",
                "mango key lime cheesecake",
                "BananaCreamCheesecake",
                "Social_WhiteChocolateCaramelMacadamiaNutCheesecake",
                "lemon raspberry cream cheesecake",
                "Chocolate Tuxedo Cheesecake",
                "Hersheys Cheesecake",
                "30thChocolateAnnivCheesecake",
                "Social_VanillaBean",
                "Social_TiramisuCheesecake",
                "Social_ChocolateChipCookieDoughCheesecake",
                "key lime cheesecake",
                "Social_LowCarbCheesecake",
                "Social_LowCarbCheesecakeWithStrawberries",
                "caramel pecan turtle cheesecake",
                "SnickersBarCheesecake",
                "CraigsCrazyCarrotCheesecake",
                "Social_CherryCheesecake",
                "pumpkin cheesecake",
                "pumpkin pecan cheesecake",
                "peppermint bark cheesecake")

# rule: end with cheesecake, no spaces

cheesecake_slug = gsub(" ","",cheesecakes)

# download.file(paste0('https://www.thecheesecakefactory.com/assets/images/Menu-Import/CCF_',
#                      cheesecake_slug, '.jpg'), 
#               paste0('data/', cheesecake_slug, '.jpg'))

# library("jpeg")
# jj <- readJPEG(paste0('data/' ,cheesecake_slug[5], '.jpg'),native=FALSE)
# plot(0:1,0:1,type="n",ann=FALSE,axes=FALSE)
# rasterImage(jj,0,0,1,1)

scale_down = 4
x = array(dim = c(length(cheesecake_slug), 1440/scale_down, 1920/scale_down, 3))

for (i in 1:length(cheesecake_slug)){
        x[i, , , ] = readJPEG(paste0('data/', cheesecake_slug[i], '.jpg'),native=FALSE) %>%
                resize(dim(.)[1]/scale_down, dim(.)[2]/scale_down) 
                
}


x_train <- array_reshape(x[,,,1], c(nrow(x), 1440 * 1920 / scale_down / scale_down))


# library(keras)
# mnist <- dataset_mnist()
# x_train <- mnist$train$x
# x_train <- array_reshape(x_train, c(nrow(x_train), 784))

model <- keras_model_sequential() 
model %>% 
        layer_dense(units = 256, activation = 'relu', input_shape = c(172800)) 

model %>% compile(
        loss = 'mean_squared_error',
        optimizer = optimizer_rmsprop(),
        metrics = c('accuracy')
)
y_train = 1:37

history <- model %>% fit(
        x_train, y_train, 
        epochs = 30, batch_size = 128, 
        validation_split = 0.2
)
