#pragma once
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>

struct ArucoConfig {
    std::string camera_pipe;
    std::string out_pipe;
    std::string aruco_dict;
    std::string image_format;     // "gray8" or "nv12"
    std::string intrinsics;
    std::string camera_name;
    double default_size_m = 0.16;
    std::unordered_map<int,double> id_size_map;
    int downscale = 1;
    double min_marker_perimeter_rate = 0.02;
    bool corner_refinement = true;
};

inline bool load_config_yaml(const std::string& path, ArucoConfig& cfg){
    cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    if(!fs.isOpened()) return false;

    fs["camera_pipe"] >> cfg.camera_pipe;
    fs["out_pipe"] >> cfg.out_pipe;
    fs["aruco_dict"] >> cfg.aruco_dict;
    fs["image_format"] >> cfg.image_format;
    fs["intrinsics"] >> cfg.intrinsics;
    fs["camera_name"] >> cfg.camera_name;
    if(!fs["default_size_m"].empty()) fs["default_size_m"] >> cfg.default_size_m;
    if(!fs["downscale"].empty()) fs["downscale"] >> cfg.downscale;
    if(!fs["min_marker_perimeter_rate"].empty()) fs["min_marker_perimeter_rate"] >> cfg.min_marker_perimeter_rate;
    if(!fs["corner_refinement"].empty()) fs["corner_refinement"] >> cfg.corner_refinement;

    // id_size_map（stringキー→intへ）
    if(!fs["id_size_map"].empty()){
        cv::FileNode m = fs["id_size_map"];
        for(auto it = m.begin(); it != m.end(); ++it){
            int id = std::stoi((*it).name());
            double val; (*it) >> val;
            cfg.id_size_map[id] = val;
        }
    }
    return true;
}

inline double size_for_id(const ArucoConfig& cfg, int id){
    auto it = cfg.id_size_map.find(id);
    return (it!=cfg.id_size_map.end()) ? it->second : cfg.default_size_m;
}
