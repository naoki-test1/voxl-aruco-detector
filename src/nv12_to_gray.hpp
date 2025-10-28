#pragma once
#include <cstdint>
#include <vector>

// meta.width x meta.height のNV12の先頭Y面をそのまま使うなら変換不要。
// 画像パイプがNV12の場合、先頭のY平面（高さ=H, 幅=W）はグレイとして使える。
// ここでは安全のためコピー関数を置いておく。
inline void nv12_y_to_gray(const uint8_t* nv12, int width, int height, std::vector<uint8_t>& out_gray){
    out_gray.resize(width*height);
    // 先頭Y面をコピー
    std::memcpy(out_gray.data(), nv12, width*height);
}
