import numpy as np
from spiro_analysis import preprocess
from spiro_analysis import cli
import matplotlib.pyplot as plt
import matplotlib as mpl

def test_background_input_dtype_is_int16():
    
    assert 3 == preprocess.background.hello()

def test_open_metadata_background_dtype():
    background, radius, center, frames = cli.open_metadata("backgroundSubtraction")
    print(type(background[0]))
    print(background[0].shape)
    print(background[0].dtype)
    assert False

def test_background_calculation():
    mpl.rc('image', cmap='gray')
    FILE = "test_passiveInteraction"
    backgrounds = preprocess.background.open_backgrounds(FILE)
    # print(backgrounds.shape)
    frames = preprocess.background.open_frames(FILE)
    print(frames.shape)
    output = preprocess.background.background_calculation(frames, backgrounds)
    print(output.shape)
    print("got output)")
    plt.imshow(output)
    plt.show()

    assert True

    # preprocess.background.background_calculation()

if __name__ == "__main__":
    test_background_calculation()