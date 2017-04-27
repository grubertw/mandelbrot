# mandelbrot
Simple implementation of the Mandelbrot set using Rust's glium library and OpenGL 4.5 shaders.

## To compile:
$> cargo build

## To build and run:
$> cargo run

Rust's Cargo will automaticly download and build dependancies listed in Cargo.toml (as long as you have Rust installed).

## OpenGL compatiblity
Current implementation requires OpenGL 4.5.  If you are running on Windows and your graphics drivers are somewhat up-to-date, and you have a decent graphics card (say anything Geforce FX or newer), then it should run.  If you are running on OSx, then it will likely not work, as it defaults to the 2.2 compatibility profile (as of Sierra 10.12). 

It is farily easy to change the code to support older OpenGL.  Just change the top line in the .glsl files to `#version 140`.  Unfortunatly, this older version of the shader language does not suport doubles, so all uses of double must be replaced with float, and uses of `dvec` must be replaced with `vec`.  

Changing doubles to floats means the zoom capibilities of the fractal are only half as good. 

## Here's the working list of commands and keybindings:
* The window can be resized and the fractal will be redrawn to fit the window (using the same scale)
* Hold the left mouse button to shift the viewport (left, right, up, or down)
* Use mouse wheel to zoom in and out (up zooms in, down zooms out)
* press = key to increase the maximum number of iterations 
* press - key to decrease the maximum number of iterations 
* press s key to increase the rate at which the image is scaled
* press a key to decrease the rate at which the image is scaled
* press r key to increase red frequency 
* press e key to decrease red frequency
* press g key to increase green frequency
* press f key to decrease green frequency
* press b key to increase blue frequency 
* press v key to decrease blue frequency
* press y key to increase red phase 
* press t key to decrease red phase
* press j key to increase green phase
* press h key to decrease green phase
* press m key to increase blue phase 
* press n key to decrease blue phase

## A note on the current color implentation
Currently, the `sin()` function is being used to pick the colors.  The concept comes from the following article:  https://krazydad.com/tutorials/makecolors.php  In efforts to reduce 'rainbow effect', I devide the iteration count by max_iterations, then multiply by a factor (defaulting to 1.0).  Taking the sin of values from 0 to 1 makes it to a little less than 1/4th the frequancy (freq of a sin wave is 2pi), so I multiply by 1/2p to strech the values to cover a quarter of the sin wave (which gives a nice curve for increasing color from 0 to 1).  The real art comes in to play for making slight changes to this frequancy for each color (i.e. each color should use a slightly different frequency). 

For maximum color diversity, the phases for each color should be shifted by a factor of 2 (say 0 for red, 2 for green, and 4 for blue). Remember, the frequancy of a sin wave is 2pi (which is about 6), so this spaces out the colors fairly even.

### Other sitations and tributes:
I borrowed heavly from the glium online tutorial to write this, as well as several other posted fractal implementations. As to not take credit away, I also pay thanks to Tomas Sedovic's article: https://aimlesslygoingforward.com/blog/2016/09/27/rendering-mandelbrot-set-using-shaders-with-rust/

