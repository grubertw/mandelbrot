use std::str::FromStr;

use iced_wgpu::Renderer;
use iced_winit::widget::slider::{self, Slider};
use iced_winit::widget::text_input::{self, TextInput};
use iced_winit::widget::{Column, Row, Text};
use iced_winit::{Alignment, Color, Command, Element, Length, Program};


#[derive(Clone)]
pub struct Controls {
    // Values
    max_iterations: u32,
    rgb_freq: Color,
    rgb_phase: Color,

    // Controls states
    max_itrs_input: text_input::State,
    rgb_sliders: [slider::State; 6],
}

#[derive(Debug, Clone)]
pub enum Message {
    MaxIterationsChanged(u32),
    RGBFreqChanged(Color),
    RGBPhaseChanged(Color)
}

impl Controls {
    pub fn new() -> Controls {
        Controls {
            max_iterations: 100,
            rgb_freq: Color::WHITE,
            rgb_phase: Color::BLACK,
            max_itrs_input: Default::default(),
            rgb_sliders: Default::default(),
        }
    }

    pub fn max_iterations(&self) -> u32 {
        self.max_iterations
    }

    pub fn rgb_freq(&self) -> Color {
        self.rgb_freq
    }

    pub fn rgb_phase(&self) -> Color {
        self.rgb_phase
    }
}

impl Program for Controls {
    type Renderer = Renderer;
    type Message = Message;

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::MaxIterationsChanged(it) => {
                self.max_iterations = it;
            }
            Message::RGBFreqChanged(c_freq) => {
                self.rgb_freq = c_freq;
            }
            Message::RGBPhaseChanged(c_phase) => {
                self.rgb_phase = c_phase;
            }
        }

        Command::none()
    }

    fn view(&mut self) -> Element<Message, Renderer> {
        let [r_freq, g_freq, b_freq, r_phase, g_phase, b_phase] = &mut self.rgb_sliders;
        let c_freq = self.rgb_freq;
        let c_phase = self.rgb_phase;
   
        let itrs_row = Row::new()
            .align_items(Alignment::Start)
            .max_width(220)
            .push(
                Text::new("Max Iterations: ")
                    .style(Color::WHITE)
            )
            .push(
                TextInput::new(
                    &mut self.max_itrs_input,
                    "Placeholder",
                    &self.max_iterations.to_string(),
                    move |it| {
                        Message::MaxIterationsChanged(FromStr::from_str(it.as_str()).unwrap())
                    })
                    .width(Length::Units(100))
            );
        let freq_row = Row::new()
            .spacing(20)
            .max_width(400)
            .push(
                Slider::new(r_freq, 0.0..=1.0, self.rgb_freq.r, move |r| {
                    Message::RGBFreqChanged(Color {
                        r,
                        ..c_freq
                    })
                })
                .step(0.01),
            )
            .push(
                Slider::new(g_freq, 0.0..=1.0, self.rgb_freq.g, move |g| {
                    Message::RGBFreqChanged(Color {
                        g,
                        ..c_freq
                    })
                })
                .step(0.01),
            )
            .push(
                Slider::new(b_freq, 0.0..=1.0, self.rgb_freq.b, move |b| {
                    Message::RGBFreqChanged(Color {
                        b,
                        ..c_freq
                    })
                })
                .step(0.01),
            );
        let phase_row = Row::new()
            .spacing(20)
            .max_width(400)
            .push(
                Slider::new(r_phase, 0.0..=1.0, self.rgb_phase.r, move |r| {
                    Message::RGBPhaseChanged(Color {
                        r,
                        ..c_phase
                    })
                })
                .step(0.01),
            )
            .push(
                Slider::new(g_phase, 0.0..=1.0, self.rgb_phase.g, move |g| {
                    Message::RGBPhaseChanged(Color {
                        g,
                        ..c_phase
                    })
                })
                .step(0.01),
            )
            .push(
                Slider::new(b_phase, 0.0..=1.0, self.rgb_phase.b, move |b| {
                    Message::RGBPhaseChanged(Color {
                        b,
                        ..c_phase
                    })
                })
                .step(0.01),
            );

        Row::new()
            .width(Length::Fill)
            .height(Length::Fill)
            .align_items(Alignment::End)
            .push(
                Column::new()
                    .width(Length::Fill)
                    .align_items(Alignment::Start)
                    .push(Column::new()
                        .padding(10)
                        .spacing(10)
                        .push(freq_row)
                        .push(phase_row)
                        .push(itrs_row)
                    )
                )
            .into()
    }
}