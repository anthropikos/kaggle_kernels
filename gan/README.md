# README and Notes

## Download the dataset

Use the shell script in the `data/` directory to download and unzip the dataset.

You will need to have [Kaggle API](https://github.com/Kaggle/kaggle-api#api-credentials) setup. Follow [this documentation](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) to setup Kaggle API.


## Todo

- [ ] Fix the plotting utility because the ImageDataset accessor now returns a Tensor.
- [ ] Implement optimizers for each of the following: monet_generator, photo_generator, monet_discriminator, photo_discriminator.


## Future investigation

- Investigate the performance of RMSE in the CycleLoss calculation instead of the current MAE implementation (see Cycle Loss function notes for more detail).


## Questions

- ??? Consider why cycle loss is abs(real-cycled)?
- ??? Consider why identity loss is also abs(real-same_image)?


## Notes

### Generator Loss

The generator loss function takes the output of a discriminator and compare it against the most ideal case. Being that this is a loss function for the generator, the objective is set to the objective of the generator instead of the discrinimator. The generator's objective is for the output of the discriminator to be a tensor full of ones thus indicating a successful tricking of making the discriminator think that the image passed to the discriminator is indeed a real image instead of a generated image. 

The data flows from the dataset to the generator, the discriminator, and then the generator loss function. The loss thus backpropagates through this computation graph backwards to the discriminator and then the generator.

I don't want the loss to be used to update the discriminator, instead this loss is meant to inform the generator and thus only used to update the generator. This differentiation thus warrants the use of having a monet_generator or photo_generator specific optimizer to only update the respective generator.


### Cycle Loss

The create a cycled image, it is passed through two different generators. A cycled monet is created using a real monet to generate a fake photo and then use the fake photo to generate a fake monet, which we call a cycled monet. A cycled photo is created using a real photo to generate a fake monet and then use the fake monet to generate a fake photo which is a cycled photo.

The cycle loss thus compares the cycled image with the original version to calculate a metric using mean absolute error (MAE) by taking the mean of the absolute of differences.

I theorize the usage of root mean square error (RMSE) would be simialr to MAE here because the generator's output is the tanh range hence severe outliers are limited. However, the advantage of one over the other is not clear as different publication seem to debate on each of their advantages and disadvantages (Karunasingha, 2022). However, because our data range is tanh range, an unbiased error metric with even penalty would be preferred. MAE is an unbiased error metric (Brassington, 2017) with less sensitivity to outliers hence a good choice for our data range for the cycled image generator.

It would be interesting to test the performance of using RMSE instead of MAE for the cycle loss calculation and this will be a suggested future investigation.

Because the loss calculated by this loss function only involves generators, it should utilize the optimizer that optimizes for generators.

References:
Karunasingha, D. S. K. (2022). Root mean square error or mean absolute error? Use their ratio as well. Information Sciences, 585, 609–629. https://doi.org/10.1016/j.ins.2021.11.036

Brassington, G. (2017). Mean absolute error and root mean square error: Which is the better metric for assessing model performance? 3574. EGU General Assembly Conference Abstracts.
