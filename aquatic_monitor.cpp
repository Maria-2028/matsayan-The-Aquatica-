#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>

using namespace cv;
using namespace dnn;
using namespace std;
namespace fs = filesystem;

#if __cplusplus < 201703L
#error "This code requires C++17 or later"
#endif

mutex logMutex;  // For thread-safe logging

struct EnvironmentalData {
    float temperature;
    float turbidity;
    float pH;
    float salinity;
    time_t timestamp;

    string toString() const {
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&timestamp));
        return "[" + string(buffer) + "] Temp: " + to_string(temperature) + "°C | Turbidity: " +
               to_string(turbidity) + " NTU | pH: " + to_string(pH) + " | Salinity: " +
               to_string(salinity) + " ppt";
    }
};

struct DetectionResult {
    string label;
    float confidence;
    float size;
    string activity;
    time_t timestamp;

    string toString() const {
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&timestamp));
        return "[" + string(buffer) + "] " + label + " (" + to_string(confidence) + "%)" +
               (size > 0 ? " | Size: " + to_string(size) + " cm" : "") +
               (!activity.empty() ? " | Activity: " + activity : "");
    }
};

class SolarPanel {
private:
    float efficiency;
    float area;
    float currentOutput;
    bool isDaytime;

public:
    SolarPanel(float eff, float a) : efficiency(eff), area(a), currentOutput(0), isDaytime(true) {}

    void update(float sunlightIntensity) {
        currentOutput = isDaytime ? sunlightIntensity * area * efficiency : 0;
    }

    float getCurrentOutput() const { return currentOutput; }
    void setDaytime(bool daytime) { isDaytime = daytime; }
    bool getIsDaytime() const { return isDaytime; }
};

class Battery {
private:
    float capacity;
    float currentCharge;
    float maxChargeRate;

public:
    Battery(float cap, float maxRate) : capacity(cap), currentCharge(cap * 0.7f), maxChargeRate(maxRate) {}

    void charge(float power, float hours) {
        float energy = min(power, maxChargeRate) * hours;
        currentCharge = min(capacity, currentCharge + energy);
    }

    bool discharge(float power, float hours) {
        float energyNeeded = power * hours;
        if (energyNeeded <= currentCharge) {
            currentCharge -= energyNeeded;
            return true;
        }
        return false;
    }

    float getChargePercentage() const { return (currentCharge / capacity) * 100.0f; }
};

class ConveyorBelt {
private:
    bool isRunning;
    float speed;
    float powerUsage;
    mutable mutex conveyorMutex; // Added 'mutable' here

public:
    ConveyorBelt() : isRunning(false), speed(0.5f), powerUsage(150.0f) {}

    void activate() {
        lock_guard<mutex> lock(conveyorMutex);
        isRunning = true;
    }

    void stop() {
        lock_guard<mutex> lock(conveyorMutex);
        isRunning = false;
    }

    bool isActive() const {
        lock_guard<mutex> lock(conveyorMutex);
        return isRunning;
    }

    float getPowerUsage() const {
        lock_guard<mutex> lock(conveyorMutex);
        return isRunning ? powerUsage : 0;
    }

    void processWaste(const DetectionResult& waste) {
        try {
            {
                lock_guard<mutex> lock(conveyorMutex);
                if (!isRunning) isRunning = true;
            }

            cout << "[CONVEYOR] Processing " << waste.label << " (" << waste.size << " cm)" << endl;

            auto start = chrono::steady_clock::now();
            while (chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start).count() < 2) {
                this_thread::sleep_for(chrono::milliseconds(100));
                if (!isActive()) break;
            }

            cout << "[CONVEYOR] Waste deposited in collection bin" << endl;
        } catch (const exception& e) {
            cerr << "[CONVEYOR ERROR] " << e.what() << endl;
        }
    }
};

class AquaticDetector {
private:
    struct WasteData {
        string waterBodyType;
        string locationType;
        string wasteType;
        string wasteSubtype;
        string imageFileName;
        float confidence;
        float size;
        float weight;
        float temperature;
        float turbidity;
        float pH;
    };

    struct MarineData {
        string waterBodyType;
        string locationType;
        string animalType;
        string animalSpecies;
        string imageFileName;
        float confidence;
        float size;
        float weight;
        string activity;
        float temperature;
        float salinity;
        float pH;
    };

    vector<WasteData> wasteDataset;
    vector<MarineData> marineDataset;
    size_t currentWasteIndex = 0;
    size_t currentMarineIndex = 0;
    bool datasetsLoaded = false;
    mutex datasetMutex;

    void skipMetadataLines(ifstream& file) {
        string line;
        while (getline(file, line)) {
            if (line.find("> metadata.") == string::npos) {
                // Put the line back for the actual parser
                file.seekg(-(static_cast<long>(line.size()) + 1), ios::cur);
                break;
            }
        }
    }

    void loadWasteDataset(const string& csvPath) {
        ifstream file(csvPath);
        if (!file.is_open()) {
            cerr << "Failed to open waste dataset file: " << csvPath << endl;
            return;
        }

        skipMetadataLines(file);

        string line;
        getline(file, line); // Skip header

        while (getline(file, line)) {
            try {
                stringstream ss(line);
                WasteData data;
                string token;

                // Skip ID
                if (!getline(ss, token, ',')) continue;

                getline(ss, data.waterBodyType, ',');
                getline(ss, data.locationType, ',');
                getline(ss, data.wasteType, ',');
                getline(ss, data.wasteSubtype, ',');
                getline(ss, data.imageFileName, ',');

                // Handle confidence
                if (!getline(ss, token, ',')) continue;
                data.confidence = token.empty() ? 0 : stof(token);

                // Handle size
                if (!getline(ss, token, ',')) continue;
                data.size = token.empty() ? 0 : stof(token);

                // Handle weight
                if (!getline(ss, token, ',')) continue;
                data.weight = token.empty() ? 0 : stof(token);

                // Handle temperature
                if (!getline(ss, token, ',')) continue;
                data.temperature = token.empty() ? 0 : stof(token);

                // Handle turbidity
                if (!getline(ss, token, ',')) continue;
                data.turbidity = token.empty() ? 0 : stof(token);

                // Handle pH
                if (!getline(ss, token)) continue;
                data.pH = token.empty() ? 7.0f : stof(token);

                lock_guard<mutex> lock(datasetMutex);
                wasteDataset.push_back(data);
            } catch (const exception& e) {
                cerr << "Error parsing waste data line: " << line << "\nError: " << e.what() << endl;
            }
        }
        file.close();
    }

    void loadMarineDataset(const string& csvPath) {
        ifstream file(csvPath);
        if (!file.is_open()) {
            cerr << "Failed to open marine dataset file: " << csvPath << endl;
            return;
        }

        skipMetadataLines(file);

        string line;
        getline(file, line); // Skip header

        while (getline(file, line)) {
            try {
                stringstream ss(line);
                MarineData data;
                string token;

                // Skip ID
                if (!getline(ss, token, ',')) continue;

                getline(ss, data.waterBodyType, ',');
                getline(ss, data.locationType, ',');
                getline(ss, data.animalType, ',');
                getline(ss, data.animalSpecies, ',');
                getline(ss, data.imageFileName, ',');

                // Handle confidence
                if (!getline(ss, token, ',')) continue;
                data.confidence = token.empty() ? 0 : stof(token);

                // Handle size
                if (!getline(ss, token, ',')) continue;
                data.size = token.empty() ? 0 : stof(token);

                // Handle weight
                if (!getline(ss, token, ',')) continue;
                data.weight = token.empty() ? 0 : stof(token);

                // Handle activity
                if (!getline(ss, data.activity, ',')) continue;

                // Handle temperature
                if (!getline(ss, token, ',')) continue;
                data.temperature = token.empty() ? 0 : stof(token);

                // Handle salinity
                if (!getline(ss, token, ',')) continue;
                data.salinity = token.empty() ? 0 : stof(token);

                // Handle pH
                if (!getline(ss, token)) continue;
                data.pH = token.empty() ? 7.0f : stof(token);

                lock_guard<mutex> lock(datasetMutex);
                marineDataset.push_back(data);
            } catch (const exception& e) {
                cerr << "Error parsing marine data line: " << line << "\nError: " << e.what() << endl;
            }
        }
        file.close();
    }

public:
    AquaticDetector(const string& wasteDatasetPath = "waste_detection_with_images_dataset.csv",
                   const string& marineDatasetPath = "expanded_marine_animal_2_dataset.csv") {
        srand(static_cast<unsigned>(time(0)));
        loadWasteDataset(wasteDatasetPath);
        loadMarineDataset(marineDatasetPath);

        lock_guard<mutex> lock(datasetMutex);
        datasetsLoaded = !wasteDataset.empty() && !marineDataset.empty();

        if (!datasetsLoaded) {
            cerr << "Warning: One or both datasets failed to load properly" << endl;
        }
    }

    pair<vector<DetectionResult>, vector<DetectionResult>> detect(Mat& frame) {
        vector<DetectionResult> marineDetections;
        vector<DetectionResult> wasteDetections;

        {
            lock_guard<mutex> lock(datasetMutex);

            if (!marineDataset.empty()) {
                DetectionResult marine;
                const auto& data = marineDataset[currentMarineIndex];
                marine.label = data.animalSpecies;
                marine.confidence = data.confidence;
                marine.size = data.size;
                marine.activity = data.activity;
                marine.timestamp = chrono::system_clock::to_time_t(chrono::system_clock::now());
                marineDetections.push_back(marine);

                currentMarineIndex = (currentMarineIndex + 1) % marineDataset.size();
            }

            if (!wasteDataset.empty()) {
                DetectionResult waste;
                const auto& data = wasteDataset[currentWasteIndex];
                waste.label = data.wasteType + (data.wasteSubtype.empty() ? "" : " (" + data.wasteSubtype + ")");
                waste.confidence = data.confidence;
                waste.size = data.size;
                waste.activity = "";
                waste.timestamp = chrono::system_clock::to_time_t(chrono::system_clock::now());
                wasteDetections.push_back(waste);

                currentWasteIndex = (currentWasteIndex + 1) % wasteDataset.size();
            }
        }

        return {marineDetections, wasteDetections};
    }

    EnvironmentalData readEnvironmentalSensors() {
        EnvironmentalData data;
        data.timestamp = chrono::system_clock::to_time_t(chrono::system_clock::now());

        lock_guard<mutex> lock(datasetMutex);
        if (!wasteDataset.empty() && !marineDataset.empty()) {
            const auto& wasteData = wasteDataset[currentWasteIndex];
            const auto& marineData = marineDataset[currentMarineIndex];

            data.temperature = (wasteData.temperature + marineData.temperature) / 2.0f;
            data.turbidity = wasteData.turbidity;
            data.pH = (wasteData.pH + marineData.pH) / 2.0f;
            data.salinity = marineData.salinity;
        } else {
            // Fallback to random data if datasets not loaded
            data.temperature = 20.0f + static_cast<float>(rand() % 15);
            data.turbidity = static_cast<float>(rand() % 50);
            data.pH = 6.5f + static_cast<float>(rand() % 5) / 2.0f;
            data.salinity = (rand() % 3 == 0) ? 0.5f : (rand() % 3 == 1) ? 15.0f : 35.0f;
        }

        return data;
    }

    bool captureFrame(Mat& frame, bool isMarine) {
        if (!frame.empty()) {
            frame.release();
        }

        string imageFile;
        {
            lock_guard<mutex> lock(datasetMutex);
            if (isMarine && !marineDataset.empty()) {
                imageFile = marineDataset[currentMarineIndex].imageFileName;
            } else if (!isMarine && !wasteDataset.empty()) {
                imageFile = wasteDataset[currentWasteIndex].imageFileName;
            } else {
                return false;
            }
        }

        Mat loadedFrame = imread(imageFile, IMREAD_COLOR);
        if (loadedFrame.empty()) {
            cerr << "Error loading image: " << imageFile << endl;
            return false;
        }

        frame = loadedFrame.clone();
        return true;
    }
};

class FloatingAquaticMonitor {
private:
    SolarPanel solarPanel;
    Battery battery;
    AquaticDetector detector;
    ConveyorBelt conveyor;
    bool isRunning;
    float detectionInterval;
    ofstream dataLog;
    mutex logMutex;

    const float CAMERA_POWER = 5.0f;
    const float PROCESSING_POWER = 10.0f;
    const float SENSOR_POWER = 2.0f;

    void logData(const string& message) {
        lock_guard<mutex> lock(logMutex);
        time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        char buffer[80];
        if (strftime(buffer, sizeof(buffer), "[%Y-%m-%d %H:%M:%S]", localtime(&now)) == 0) {
            strcpy(buffer, "[Invalid Time]");
        }

        dataLog << buffer << " " << message << endl;
        cout << buffer << " " << message << endl;
    }

    string formatTime(time_t time) {
        char buffer[80];
        if (strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S]", localtime(&time)) == 0) {
            return "Invalid Time";
        }
        return string(buffer);
    }

public:
    FloatingAquaticMonitor()
        : solarPanel(0.20f, 0.75f),
          battery(500.0f, 100.0f),
          isRunning(false),
          detectionInterval(1.0f / 6.0f) {

        dataLog.open("aquatic_monitor_log.txt", ios::app);
        if (!dataLog.is_open()) {
            throw runtime_error("Failed to open log file");
        }
    }

    ~FloatingAquaticMonitor() {
        stop();
        if (dataLog.is_open()) {
            dataLog.flush();
            dataLog.close();
        }
    }

    bool initialize() {
        logData("System initialized");
        return true;
    }

    void run() {
        isRunning = true;
        logData("Starting marine life and waste monitoring system");

        auto lastDetectionTime = chrono::system_clock::now();
        auto lastEnvReadingTime = lastDetectionTime;
        vector<DetectionResult> lastMarineDetections;
        vector<DetectionResult> lastWasteDetections;
        EnvironmentalData lastEnvData;

        while (isRunning && battery.getChargePercentage() > 5.0f) {
            try {
                auto currentTime = chrono::system_clock::now();
                float elapsedHours = chrono::duration<float>(currentTime - lastDetectionTime).count() / 3600.0f;

                time_t now = chrono::system_clock::to_time_t(currentTime);
                tm localTime = *localtime(&now);
                bool isDay = localTime.tm_hour >= 6 && localTime.tm_hour < 18;
                solarPanel.setDaytime(isDay);

                float sunlightIntensity = isDay ? 500.0f + 300.0f * sin((localTime.tm_hour - 6) * M_PI / 12.0f) : 0.0f;
                solarPanel.update(sunlightIntensity);
                battery.charge(solarPanel.getCurrentOutput(), elapsedHours);

                float envElapsedHours = chrono::duration<float>(currentTime - lastEnvReadingTime).count() / 3600.0f;
                if (envElapsedHours >= 1.0f / 12.0f) {
                    if (battery.discharge(SENSOR_POWER, 0.01f)) {
                        lastEnvData = detector.readEnvironmentalSensors();
                        logData("Environmental Data: " + lastEnvData.toString());

                        if (lastEnvData.pH < 6.5 || lastEnvData.pH > 8.5) {
                            logData("WARNING: Critical pH level detected!");
                        }
                        if (lastEnvData.turbidity > 50.0f) {
                            logData("WARNING: High turbidity detected!");
                        }

                        lastEnvReadingTime = currentTime;
                    }
                }

                if (elapsedHours >= detectionInterval) {
                    if (battery.discharge(CAMERA_POWER + PROCESSING_POWER, 0.05f)) {
                        Mat frame;
                        bool marineFrame = (rand() % 2 == 0);
                        if (detector.captureFrame(frame, marineFrame)) {
                            auto [marineDetections, wasteDetections] = detector.detect(frame);

                            if (!marineDetections.empty()) {
                                logData("Marine Life Detected:");
                                for (const auto& detection : marineDetections) {
                                    logData("-> " + detection.toString());
                                }
                            }

                            if (!wasteDetections.empty()) {
                                logData("Waste Detected:");
                                for (const auto& waste : wasteDetections) {
                                    logData("-> " + waste.toString());

                                    if (battery.getChargePercentage() > 20.0f) {
                                        conveyor.processWaste(waste);
                                        battery.discharge(conveyor.getPowerUsage(), 2.0f / 3600.0f);
                                    } else {
                                        logData("Low battery - skipping waste collection");
                                    }
                                }
                            }

                            if (marineDetections.empty() && wasteDetections.empty()) {
                                logData("No objects detected");
                            }

                            lastMarineDetections = marineDetections;
                            lastWasteDetections = wasteDetections;
                            lastDetectionTime = currentTime;
                        }
                    } else {
                        logData("Low battery - skipping detection cycle");
                    }
                }

                static auto lastStatusTime = currentTime;
                float statusElapsedHours = chrono::duration<float>(currentTime - lastStatusTime).count() / 3600.0f;
                if (statusElapsedHours >= 0.25f) {
                    stringstream status;
                    status << "===== SYSTEM STATUS =====" << endl;
                    status << "Time: " << formatTime(now) << endl;
                    status << "Solar Output: " << solarPanel.getCurrentOutput() << " W" << endl;
                    status << "Battery Level: " << battery.getChargePercentage() << "%" << endl;
                    status << "Mode: " << (solarPanel.getIsDaytime() ? "Day" : "Night") << endl;

                    if (!lastMarineDetections.empty()) {
                        status << "Last Marine Detection: " << lastMarineDetections[0].label
                               << " (" << lastMarineDetections[0].confidence << "%)" << endl;
                    }

                    if (!lastWasteDetections.empty()) {
                        status << "Last Waste Detection: " << lastWasteDetections[0].label
                               << " (" << lastWasteDetections[0].size << " cm)" << endl;
                    }

                    status << "Environment: " << lastEnvData.temperature << "°C, "
                           << lastEnvData.turbidity << " NTU, pH " << lastEnvData.pH << endl;
                    status << "========================";

                    logData(status.str());
                    lastStatusTime = currentTime;
                }

                this_thread::sleep_for(chrono::seconds(1));
            } catch (const exception& e) {
                logData(string("ERROR in main loop: ") + e.what());
                this_thread::sleep_for(chrono::seconds(1)); // Prevent tight error loop
            }
        }

        if (battery.getChargePercentage() <= 5.0f) {
            logData("CRITICAL: Battery level below 5% - initiating shutdown");
        }
    }

    void stop() {
        isRunning = false;
        conveyor.stop();
        logData("System shutdown complete");
    }
};

int main() {
    try {
        FloatingAquaticMonitor monitor;

        if (!monitor.initialize()) {
            cerr << "Failed to initialize monitoring system" << endl;
            return 1;
        }

        thread monitorThread([&monitor]() {
            try {
                monitor.run();
            } catch (const exception& e) {
                cerr << "Monitor thread exception: " << e.what() << endl;
            }
        });

        this_thread::sleep_for(chrono::minutes(5));

        monitor.stop();
        if (monitorThread.joinable()) {
            monitorThread.join();
        }
    } catch (const exception& e) {
        cerr << "Main exception: " << e.what() << endl;
        return 1;
    }
    return 0;
}
