use super::scene::Scene;

use iced_wgpu::Renderer;
use iced_widget::{slider, text_input, column, row, text};
use iced_widget::core::{Alignment, Color, Element, Theme, Length};
use iced_winit::runtime::{Program, Task};

use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone)]
pub struct Controls {
    max_iterations: u32,
    rgb_freq: Color,
    rgb_phase: Color,
    scene: Rc<RefCell<Scene>>
}

#[derive(Debug, Clone)]
pub enum Message {
    MaxIterationsChanged(String),
    RGBFreqChanged(Color),
    RGBPhaseChanged(Color)
}

impl Controls {
    pub fn new(s: Rc<RefCell<Scene>>) -> Controls {
        Controls {
            max_iterations: 100,
            rgb_freq: Color::WHITE,
            rgb_phase: Color::BLACK,
            scene: s
        }
    }
}

impl Program for Controls {
    type Theme = Theme;
    type Message = Message;
    type Renderer = Renderer;

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::MaxIterationsChanged(it_str) => {
                if let Ok(it) = it_str.parse::<u32>() {
                    self.max_iterations = it;
                    self.scene.borrow_mut().set_max_iterations(it);
                }
            }
            Message::RGBFreqChanged(c_freq) => {
                self.rgb_freq = c_freq;
                self.scene.borrow_mut().set_rgb_freq((c_freq.r, c_freq.g, c_freq.b));
            }
            Message::RGBPhaseChanged(c_phase) => {
                self.rgb_phase = c_phase;
                self.scene.borrow_mut().set_rgb_phase((c_phase.r, c_phase.g, c_phase.b));
            }
        }

        Task::none()
    }

    fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        let c_freq = self.rgb_freq;
        let c_phase = self.rgb_phase;
   
        let itrs_row = row![
            text("Max Iterations: ").color(Color::WHITE),
            text_input("Placeholder...", &self.max_iterations.to_string())
                .on_input(Message::MaxIterationsChanged)
                .width(Length::Fixed(100.0))
        ]
        .align_y(Alignment::Start)
        .width(220);

        let freq_row = row![
            slider(0.0..=1.0, c_freq.r,  
                    move |r| Message::RGBFreqChanged(Color {r, ..c_freq}))
                .step(0.01),
            slider(0.0..=1.0, c_freq.g,  
                    move |g| Message::RGBFreqChanged(Color {g, ..c_freq}))
                .step(0.01),
            slider(0.0..=1.0, c_freq.b,  
                    move |b| Message::RGBFreqChanged(Color {b, ..c_freq}))
                .step(0.01),
        ]
        .spacing(20)
        .width(400);

        let phase_row = row![
            slider(0.0..=1.0, c_phase.r,  
                    move |r| Message::RGBPhaseChanged(Color {r, ..c_phase}))
                .step(0.01),
            slider(0.0..=1.0, c_phase.g,  
                    move |g| Message::RGBPhaseChanged(Color {g, ..c_phase}))
                .step(0.01),
            slider(0.0..=1.0, c_phase.b,  
                    move |b| Message::RGBPhaseChanged(Color {b, ..c_phase}))
                .step(0.01),
        ]
        .spacing(20)
        .width(400);

        row![
            column![
                column![freq_row, phase_row, itrs_row]
                .padding(10)
                .spacing(10)
            ]
            .width(Length::Fill)
            .align_x(Alignment::Start)
        ]
        .width(Length::Fill)
        .height(Length::Fill)
        .align_y(Alignment::End)
        .into()
    }
}