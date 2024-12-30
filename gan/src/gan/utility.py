# Anthony Lee 2024-12-29

def calculate_conv_output(input_size:int, kernel_size:int, padding:int, stride:int):
    input_size = input_size + 2*padding  # Assume same padding size before and after
    single_pixel_space = input_size - (kernel_size - 1)
    output_size = int( (single_pixel_space - 1) // stride )  # Calculate the floor
    output_size = output_size + 1  # The previously removed one
    return output_size

