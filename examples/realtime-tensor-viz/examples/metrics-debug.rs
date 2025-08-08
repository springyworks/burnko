#[cfg(feature = "metrics-sysinfo")]
use sysinfo::{System, CpuRefreshKind, RefreshKind};
#[cfg(feature = "metrics-nvml")]
use nvml_wrapper::{Nvml, enum_wrappers::device::TemperatureSensor};

fn main() {
    println!("Metrics debug starting...");

    #[cfg(feature = "metrics-sysinfo")]
    {
        let mut sys = System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
        sys.refresh_cpu();
        let cpus = sys.cpus();
        if cpus.is_empty() {
            println!("sysinfo: no CPUs reported");
        } else {
            let avg: f32 = cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / (cpus.len() as f32);
            println!("CPU avg usage: {:.1}% ({} cores)", avg, cpus.len());
        }
    }

    #[cfg(feature = "metrics-nvml")]
    {
        match Nvml::init() {
            Ok(nvml) => match nvml.device_by_index(0) {
                Ok(device) => {
                    match device.utilization_rates() {
                        Ok(util) => println!("GPU util: {}%", util.gpu),
                        Err(e) => println!("GPU util error: {e:?}"),
                    }
                    match device.temperature(TemperatureSensor::Gpu) {
                        Ok(t) => println!("GPU temp: {}°C", t),
                        Err(e) => println!("GPU temp error: {e:?}"),
                    }
                }
                Err(e) => println!("NVML device 0 error: {e:?}"),
            },
            Err(e) => println!("NVML init error: {e:?}"),
        }
    }

    #[cfg(not(feature = "metrics-sysinfo"))]
    println!("sysinfo feature not enabled");
    #[cfg(not(feature = "metrics-nvml"))]
    println!("nvml feature not enabled");
}
