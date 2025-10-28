#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <modal_pipe_client.h>
#include <modal_pipe_server.h>
#include <modal_pipe_interfaces.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <string>
#include "config.hpp"
#include "nv12_to_gray.hpp"

static cv::Ptr<cv::aruco::Dictionary> dict_from_name(const std::string& name){
    using namespace cv::aruco;
    static const std::unordered_map<std::string, PREDEFINED_DICTIONARY_NAME> map = {
        {"DICT_4X4_50", DICT_4X4_50}, {"DICT_4X4_100", DICT_4X4_100},
        {"DICT_5X5_50", DICT_5X5_50}, {"DICT_5X5_100", DICT_5X5_100},
        {"DICT_6X6_50", DICT_6X6_50}, {"DICT_6X6_100", DICT_6X6_100},
    };
    auto it = map.find(name);
    return cv::aruco::getPredefinedDictionary(it==map.end()? cv::aruco::DICT_4X4_50 : it->second);
}

int main(int argc, char** argv){
    // 1) 設定読み込み
    std::string cfg_path = (argc>=2)? argv[1] : "/etc/modalai/voxl-aruco-detector.conf.yaml";
    ArucoConfig cfg;
    if(!load_config_yaml(cfg_path, cfg)){
        fprintf(stderr, "Failed to load config: %s\n", cfg_path.c_str());
        return 1;
    }

    // 2) カメラ内部パラ
    cv::Mat K, D;
    {
        cv::FileStorage fs(cfg.intrinsics, cv::FileStorage::READ);
        fs["camera_matrix"] >> K;
        fs["distortion_coefficients"] >> D;
        fs.release();
        if(K.empty() || D.empty()){
            fprintf(stderr, "intrinsics not found: %s\n", cfg.intrinsics.c_str());
            return 1;
        }
    }

    // 3) MPA 入力
    int img_ch=-1;
    if(pipe_client_open_path(&img_ch, cfg.camera_pipe.c_str(), "voxl-aruco-detector", EN_PIPE_CLIENT_SIMPLE_HELPER, 0)){
        fprintf(stderr, "failed to open image pipe: %s\n", cfg.camera_pipe.c_str());
        return 1;
    }

    // 4) 出力サーバ
    int srv=-1;
    if(pipe_server_create(&srv, cfg.out_pipe.c_str(), sizeof(tag_detection_t), "voxl-aruco-detector")){
        fprintf(stderr, "failed to create out pipe: %s\n", cfg.out_pipe.c_str());
        return 1;
    }

    // 5) ArUco 検出器設定
    auto dict = dict_from_name(cfg.aruco_dict);
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    params->minMarkerPerimeterRate = cfg.min_marker_perimeter_rate;
    params->cornerRefinementMethod = cfg.corner_refinement
        ? cv::aruco::CORNER_REFINE_SUBPIX : cv::aruco::CORNER_REFINE_NONE;

    // ループ
    while(true){
        image_metadata_t meta;
        if(pipe_client_read(img_ch, &meta, sizeof(meta)) != (ssize_t)sizeof(meta)){
            usleep(1000);
            continue;
        }
        std::vector<uint8_t> imgbuf(meta.size_bytes);
        if(pipe_client_read(img_ch, imgbuf.data(), meta.size_bytes) != (ssize_t)meta.size_bytes){
            continue;
        }

        // 6) 画像をGrayへ（gray8 or nv12）
        cv::Mat gray;
        if(cfg.image_format == "gray8"){
            gray = cv::Mat(meta.height, meta.width, CV_8UC1, imgbuf.data()).clone();
        } else if(cfg.image_format == "nv12"){
            std::vector<uint8_t> y;
            nv12_y_to_gray(imgbuf.data(), meta.width, meta.height, y);
            gray = cv::Mat(meta.height, meta.width, CV_8UC1, y.data()).clone();
        } else {
            fprintf(stderr, "Unsupported image_format: %s\n", cfg.image_format.c_str());
            continue;
        }

        // optional downscale（高速化）
        cv::Mat proc = gray;
        double scale = 1.0;
        if(cfg.downscale > 1){
            cv::resize(gray, proc, cv::Size(), 1.0/cfg.downscale, 1.0/cfg.downscale, cv::INTER_AREA);
            scale = (double)cfg.downscale;
        }

        // 7) 検出
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(proc, dict, corners, ids, params);

        // 8) PnP
        for(size_t i=0;i<ids.size();++i){
            int id = ids[i];
            // スケール戻す
            std::vector<cv::Point2f> c = corners[i];
            if(scale != 1.0){
                for(auto& p : c){ p.x *= scale; p.y *= scale; }
            }

            double size_m = size_for_id(cfg, id);
            double s = size_m * 0.5;
            std::vector<cv::Point3f> obj = {
                {-s,  s, 0}, { s,  s, 0}, { s, -s, 0}, {-s, -s, 0},
            };

            cv::Vec3d rvec, tvec;
            // IPPE_SQUARE は平面正方タグに強い
            cv::solvePnP(obj, c, K, D, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
            cv::Mat R33; cv::Rodrigues(rvec, R33);

            // 9) 配信
            tag_detection_t det = {};
            det.id = id;
            det.size_m = (float)size_m;
            det.timestamp_ns = meta.timestamp_ns;
            snprintf(det.name, sizeof(det.name), "default_name");
            snprintf(det.cam,  sizeof(det.cam),  "%s", cfg.camera_name.c_str());
            det.T_tag_wrt_cam[0] = (float)tvec[0];
            det.T_tag_wrt_cam[1] = (float)tvec[1];
            det.T_tag_wrt_cam[2] = (float)tvec[2];
            for(int r=0;r<3;++r) for(int cc=0;cc<3;++cc)
                det.R_tag_to_cam[r][cc] = (float)R33.at<double>(r,cc);

            pipe_server_write(srv, &det, sizeof(det));
        }
    }

    pipe_client_close(img_ch);
    pipe_server_close(srv);
    return 0;
}
