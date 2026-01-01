use super::scene::Scene;

use iced_wgpu::Renderer;
use iced_widget::{text_input, column, row, text};
use iced_widget::core::{Alignment, Color, Element, Theme, Length};
use iced_winit::runtime::{Program, Task};

use log::{trace};

use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone)]
pub struct Controls {
    max_iterations: u32,
    debug_msg: String,
    scene: Rc<RefCell<Scene>>
}

#[derive(Debug, Clone)]
pub enum Message {
    MaxIterationsChanged(String),
    UpdateDebugText(String),
}

impl Controls {
    pub fn new(s: Rc<RefCell<Scene>>) -> Controls {
        Controls {
            max_iterations: 500,
            debug_msg: "debug info loading...".to_string(),
            scene: s
        }
    }
}

impl Program for Controls {
    type Theme = Theme;
    type Message = Message;
    type Renderer = Renderer;

    fn update(&mut self, message: Message) -> Task<Message> {
        trace!("Update {:?}", message);
        match message {
            Message::MaxIterationsChanged(it_str) => {
                if let Ok(it) = it_str.parse::<u32>() {
                    self.max_iterations = it;
                    self.scene.borrow_mut().set_max_iterations(it);
                }
            }
            Message::UpdateDebugText(dbg_msg) => {
                self.debug_msg = dbg_msg;
            }
        }

        Task::none()
    }

    fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        trace!("View");

        let itrs_row = row![
            text("Max Iterations: ").color(Color::WHITE),
            text_input("Placeholder...", &self.max_iterations.to_string())
                .on_input(Message::MaxIterationsChanged)
                .width(Length::Fixed(100.0))
        ]
        .align_y(Alignment::Start)
        .width(220);

        let dbg_row = row![
            text(&self.debug_msg).color(Color::WHITE)
                .size(10)
                .width(Length::Fixed(500.0))
        ]
        .align_y(Alignment::Start);

        row![
            column![
                column![itrs_row, dbg_row]
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