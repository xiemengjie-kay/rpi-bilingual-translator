import sounddevice as sd

def check_device_sample_rates(device_index):
    try:
        device_info = sd.query_devices(device_index)
        print(f"--- Device {device_index} Info ---")
        for key, value in device_info.items():
            print(f"{key}: {value}")

        if device_info['max_input_channels'] > 0:
            print("\n--- Testing Supported Input Sample Rates ---")
            test_rates = [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000, 192000]
            supported_input_rates = []
            for rate in test_rates:
                try:
                    with sd.InputStream(samplerate=rate, channels=device_info['max_input_channels'], device=device_index):
                        supported_input_rates.append(rate)
                    print(f"  {rate} Hz: SUPPORTED")
                except sd.PortAudioError:
                    print(f"  {rate} Hz: NOT SUPPORTED")
                except Exception as e:
                    print(f"  {rate} Hz: Error - {e}")
            print(f"\nSummary of Supported Input Rates for Device {device_index}: {supported_input_rates}")
        else:
            print(f"\nDevice {device_index} has no input channels.")

    except sd.PortAudioError as e:
        print(f"Error querying device {device_index}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    microphone_device_index = 1
    check_device_sample_rates(microphone_device_index)

    print("\n--- All Devices Overview ---")
    print(sd.query_devices())