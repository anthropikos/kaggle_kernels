# %% [markdown]
# # Model v4
# Anthony Lee 2025-01-03
# 
# Improvement TODO:
# - Fixed loss function 
# 

# %%
from gan.gan import CycleGAN, train_one_epoch, checkpoint_save
from gan.data import ImageDataset, ImageDataLoader
import torch
from pprint import pprint
from gan.plotting_utility import plot_before_after
from gan.data_processing import map_tanh_to_rgb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print( torch.cuda.get_device_properties(device) )


monet_dataset = ImageDataset(data_dir="../data/gan-getting-started/monet_jpg")
photo_dataset = ImageDataset(data_dir="../data/gan-getting-started/photo_jpg")

monet_dataloader = ImageDataLoader(monet_dataset)
photo_dataloader = ImageDataLoader(photo_dataset)


# %%
model = CycleGAN()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
epoch_count = 50
for idx_epoch in range(epoch_count):

    loss_tracker = train_one_epoch(monet_dataloader=monet_dataloader, 
                    photo_dataloader=photo_dataloader,
                    optimizer=optimizer,
                    model=model, 
                    device=device)
    
    ## Plot an example image
    idx = 20

    # Photo to monet
    input = photo_dataset[idx].to(device=device)
    output = model.generate_monet(input)  # Need to move to CPU?
    output = map_tanh_to_rgb(output)
    fig1 = plot_before_after(input, output, suptitle=f"Epoch {idx_epoch}: Photo -> Monet")
    # fig1.show()

    # Monet to photo
    input = monet_dataset[idx].to(device=device)
    output = model.generate_photo(input) # Need to move to CPU?
    output = map_tanh_to_rgb(output)
    fig2 = plot_before_after(input, output, suptitle=f"Epoch {idx_epoch}: Monet -> Photo")
    # fig2.show()

    ## Checkpoint
    if idx_epoch % 5 == 0:
        fig1.savefig(f"./epoch_{idx_epoch}_fake_monet.jpg")
        fig2.savefig(f"./epoch_{idx_epoch}_fake_photo.jpg")
        _ = checkpoint_save(epoch=idx_epoch, 
                    save_path=".", 
                    model=model,
                    optimizer=optimizer,
                    loss_tracker=loss_tracker)