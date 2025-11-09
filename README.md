# mandelbrot (2.0)
Impementation of Mandelbrot fractal using Rust's Iced GUI Library.

I try to get fancy by creating an 'overlay' of widets on top the fractal, similar to how controls are overlayed within a graphics engine. This is accomplished using Iced's underlying iced_winit and iced_wgpu.

## To compile:
$> cargo build

## To build and run:
$> cargo run

## A note on the current color implentation
Currently, the `sin()` function is being used to pick the colors.  The concept comes from the following article:  https://krazydad.com/tutorials/makecolors.php  In efforts to reduce 'rainbow effect', I devide the iteration count by max_iterations, then multiply by a factor (defaulting to 1.0).  Taking the sin of values from 0 to 1 makes it to a little less than 1/4th the frequancy (freq of a sin wave is 2pi), so I multiply by 1/2p to strech the values to cover a quarter of the sin wave (which gives a nice curve for increasing color from 0 to 1).  The real art comes in to play for making slight changes to this frequancy for each color (i.e. each color should use a slightly different frequency). 

For maximum color diversity, the phases for each color should be shifted by a factor of 2 (say 0 for red, 2 for green, and 4 for blue). Remember, the frequancy of a sin wave is 2pi (which is about 6), so this spaces out the colors fairly even.

### Other sitations and tributes:
Here are the examples/tutorials I followed to help me write this (2.0) version of Mandelbrot:

https://github.com/iced-rs/iced/tree/0.13.1/examples/integration


NOTE: As with all exaples in Git, make sure to view the correct code that has been tagged with the correct release of Iced. Also note, Iced seems to be good at keeping breaking changes from accouring within incremental releases; i.e. all iced_* libs in the 0.13.x series should be compatible. Using a never verion of wgpu might be possible, but only if it's API updates are compatible with iced_wgpu; so it's probably not a good idea.

https://sotrh.github.io/learn-wgpu/#what-is-wgpu

# Screenshots
![Mandelbrot 2.0 800x600 - screenshot 1](Mandelbrot_2_ss1.png)