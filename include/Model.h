#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <algorithm>
#include <map>

struct descriptor
{
    cv::Point loc;
    unsigned char magn;
    unsigned char phase; // quantized
};

struct score_info
{
    int index_view;
    float max_score;
    int x;
    int y;
};

class Model
{

public:
    Model(std::string datasetPath, std::string datasetName);

    std::string getPath();

    std::string getName();

    void start();

    // load models
    int loadViews();

    // load masks
    int loadMasks();

    // load masks
    int loadTestImages();

    void extractKeypointsFromViews();

    // void compute binarized phase images
    void computePhaseImages();

    void computeResponseMaps();

    int computeAllScores();

    float computeScore(int index_image, int index_view, cv::Point location);

    void initReferencePhases();

    int writeScores();

protected:
    std::string datasetPath;
    std::string datasetName;
    std::vector<std::map<u_char, float>> response_maps;
    std::vector<float> reference_phases_rad;

    std::vector<cv::Mat> views;                       // Vector of views
    std::vector<std::string> name_views;              // views filenames corresponding to the views vector
    std::vector<cv::Mat> masks;                       // Vector of masks
    std::vector<cv::Mat> test_images;                 // Vector of test images
    std::vector<std::string> name_test_images;        // test images filenames corresponding to test_images vector
    std::vector<std::vector<descriptor>> descriptors; // Vector of descriptors for each view
    std::vector<cv::Mat> phaseMaps;                   // Vector of the gradient phase maps associated to the test_images
    std::vector<std::vector<score_info>> scores;      // A list of score_info for each test image
};

// External functions

void getDiscriminativeLocations(cv::Mat &view, std::vector<descriptor> &good_descriptors, int th);

void computeMostFreqPhaseRGB(cv::Mat &magnitude, cv::Mat &phase, cv::Mat &magnitude_refined, cv::Mat &phase_refined, int th);

void computeGradientImage(cv::Mat &input_img, cv::Mat &magnitude, cv::Mat &phase);

void selectStrongestPhase(std::vector<cv::Mat> &magnitude_BGR, std::vector<cv::Mat> &phase_BGR, cv::Mat &output_magnitude, cv::Mat &output_phase, bool quantize);

void spreadPhase(cv::Mat &input_phase, cv::Mat &output_phase_spread, int wind_size, bool binary_count, bool quantize);

void increaseCounterPhase(std::vector<cv::Mat> &vector_counters, int row, int col, unsigned char phase, int wind_size, bool binary_count);

void computeMostFreqPhase(std::vector<cv::Mat> &vector_counters, cv::Mat &output_phase);

// quantize phase
unsigned char quantizePhase(float phase);

// utility comparator function to pass to the sort() module
bool sortByMagn(const descriptor &a,
                const descriptor &b);

// utility comparator function to pass to the sort() module
bool sortByScore(const score_info &a,
                 const score_info &b);