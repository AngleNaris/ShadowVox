#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "signalsmith-stretch.h"

struct Point {
    double first;
    double second;
    double third;
};

static double interpolateSecond(const std::vector<Point>& points, double at) {
    if (at <= points.front().first) return points.front().second;
    if (at >= points.back().first) return points.back().second;
    auto next = std::upper_bound(points.begin(), points.end(), at,
        [](double at, const Point& point) { return at < point.first; });
    const auto& prev = *(next - 1);
    double ratio = (at - prev.first) / (next->first - prev.first);
    return prev.second + ratio * (next->second - prev.second);
}

static double interpolateThird(const std::vector<Point>& points, double at) {
    if (at <= points.front().first) return points.front().third;
    if (at >= points.back().first) return points.back().third;
    auto next = std::upper_bound(points.begin(), points.end(), at,
        [](double at, const Point& point) { return at < point.first; });
    const auto& prev = *(next - 1);
    double ratio = (at - prev.first) / (next->first - prev.first);
    return prev.third + ratio * (next->third - prev.third);
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("usage: stretch pitch|time sr f0 in.f32 map.txt out.f32");
        const std::string mode = argv[1];
        if (mode != "pitch" && mode != "time") throw std::runtime_error("invalid render mode");
        const bool pitch = mode == "pitch";
        const int sr = std::stoi(argv[2]);
        if (sr < 1000) throw std::runtime_error("invalid sample rate");
        const float defaultF0 = std::stof(argv[3]);
        auto from_ms = [sr](double ms) { return int(std::lround(sr * ms / 1000)); };
        std::ifstream input(argv[4], std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("cannot open input samples");
        const int inputLength = int(input.tellg()) / sizeof(float);
        std::ifstream mapFile(argv[5]);
        std::vector<Point> map;
        bool hasLocalF0 = false;
        std::string line;
        while (std::getline(mapFile, line)) {
            std::istringstream iss(line);
            double a = 0., b = 0., c = 0.;
            if (!(iss >> a >> b)) continue;
            if (iss >> c) {
                hasLocalF0 = true;
            } else {
                c = defaultF0;
            }
            map.push_back(Point{a, b, c});
        }
        if (map.size() < 2 || inputLength <= 0) throw std::runtime_error("empty input or map");
        const int outputLength = pitch ? inputLength : int(std::lround(map.back().first));
        const int sourceEnd = pitch ? inputLength : int(std::lround(map.back().second));
        const int pad = from_ms(350);
        std::vector<float> source(inputLength + 2*pad);
        input.seekg(0);
        if (!input.read(reinterpret_cast<char*>(source.data() + pad), inputLength * sizeof(float))) {
            throw std::runtime_error("short input read");
        }
        signalsmith::stretch::SignalsmithStretch<float> engine(1);
        engine.configure(1, from_ms(60), from_ms(10), false);
        const int inLatency = engine.inputLatency(), outLatency = engine.outputLatency();
        const float tonalityLimit = std::min(0.45f, 8000.0f / float(sr));
        if (pitch) {
            engine.setFormantFactor(1.f, true);
            engine.setFormantBase(defaultF0 / float(sr));
        }
        auto inputAt = [&](int t) {
            if (pitch || t <= 0) return t;
            if (t >= outputLength) return sourceEnd + t - outputLength;
            return int(std::lround(interpolateSecond(map,t)));
        };
        const int stop = outputLength + outLatency;
        std::vector<float> target(stop + pad);
        const int controlBlock = from_ms(5);
        // Automation uses the synthesis timeline. Input leads by the analysis latency.
        // Keep both sides padded, then drop the output latency to restore exact placement.
        // Never crossfade phase-vocoder wet audio against dry audio mid-phrase: phase drift causes comb-filter dips.
        for (int pos = -pad; pos < stop;) {
            const int next = std::min(stop, pos + controlBlock);
            if (pitch) {
                const int centre = (pos + next) / 2;
                float cents = centre >= 0 && centre < outputLength ? float(interpolateSecond(map,centre)) : 0.f;
                if (hasLocalF0 && centre >= 0 && centre < outputLength) {
                    float localF0 = float(interpolateThird(map, centre));
                    if (localF0 > 40.f) engine.setFormantBase(localF0 / float(sr));
                }
                engine.setTransposeSemitones(cents / 100.f, tonalityLimit);
            }
            const int begin = inputAt(pos) + inLatency + pad;
            const int end = inputAt(next) + inLatency + pad;
            if (begin < 0 || end < begin || end > int(source.size())) throw std::runtime_error("invalid audio bounds");
            const float* in[] = {source.data() + begin};
            float* out[] = {target.data() + pos + pad};
            engine.process(in,end - begin,out,next - pos);
            pos = next;
        }
        std::ofstream output(argv[6],std::ios::binary);
        output.write(reinterpret_cast<const char*>(target.data() + pad + outLatency), outputLength * sizeof(float));
        if (!output) throw std::runtime_error("could not write render");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
