extern crate log;
extern crate time;

use log::{LogRecord, LogLevel, LogLevelFilter, LogMetadata, SetLoggerError};

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &LogMetadata) -> bool {
        metadata.level() <= LogLevel::Info
    }

    fn log(&self, record: &LogRecord) {
        if self.enabled(record.metadata()) {
            println!("[{}][{}] {}", 
                time::strftime("%Y-%m-%d %H:%M:%S %Z", &time::now()).unwrap(),
                record.level(), 
                record.args());
        }
    }
}

pub fn init() -> Result<(), SetLoggerError> {
    log::set_logger(|max_log_level| {
        max_log_level.set(LogLevelFilter::Info);
        Box::new(SimpleLogger)
    })
}