#include "../include/Model.h"
#include "../include/results_writer.h"

// constructors
Model::Model(std::string datasetPath, std::string datasetName)
{
    this->datasetPath = datasetPath;
    this->datasetName = datasetName;
}

// get functions

std::string Model::getPath()
{
    return datasetPath;
}

std::string Model::getName()
{
    return datasetName;
}

// Functions

void Model::start()
{
    std::cout << "### Working on the '" << datasetName << "' dataset. ###" << std::endl;
    loadViews();
    loadTestImages();
    initReferencePhases();
    extractKeypointsFromViews();
    computePhaseImages();
    computeResponseMaps();
    computeAllScores();
    writeScores();
}

int Model::loadViews()
{
    std::cout << "Loading models...\n";
    std::vector<std::string> filenames;
    // Generate all file names of the models to load
    cv::utils::fs::glob(datasetPath + "/" + datasetName + "/models", "model*", filenames);
    // Load the models
    for (int i = 0; i < filenames.size(); ++i)
    {
        cv::Mat obj = cv::imread(filenames[i]);
        if (obj.empty())
        {
            std::cout << "Model " << filenames[i] << " not loaded.\n";
            return -1;
        }
        views.push_back(obj);
        std::string name = filenames[i];
        std::size_t found = name.find_last_of("/");
        if (found != std::string::npos)
            name.erase(0, found + 1);
        name_views.push_back(name);
    }
    std::cout << "Models loaded: " << views.size() << "/" << filenames.size() << "\n\n";
    return 0;
}

int Model::loadMasks()
{
    std::cout << "Loading masks...\n";
    std::vector<std::string> filenames;
    // Generate all file names of the masks to load
    cv::utils::fs::glob(datasetPath + "/" + datasetName + "/models", "mask*", filenames);
    // Load the models
    for (int i = 0; i < filenames.size(); ++i)
    {
        cv::Mat obj = cv::imread(filenames[i], cv::IMREAD_GRAYSCALE);
        if (obj.empty())
        {
            std::cout << "Mask " << filenames[i] << " not loaded.\n";
            return -1;
        }
        masks.push_back(obj);
    }

    std::cout << "Masks loaded: " << masks.size() << "/" << filenames.size() << "\n\n";
    return 0;
}

int Model::loadTestImages()
{
    std::cout << "Loading test images...\n";
    std::vector<std::string> filenames;
    // Generate all file names of the test images to load
    cv::utils::fs::glob(datasetPath + "/" + datasetName + "/test_images", "test*", filenames);
    // Load the models
    for (int i = 0; i < filenames.size(); ++i)
    {
        cv::Mat obj = cv::imread(filenames[i]);
        if (obj.empty())
        {
            std::cout << "Test image " << filenames[i] << " not loaded.\n";
            return -1;
        }
        test_images.push_back(obj);
        std::string name = filenames[i];
        std::size_t found = name.find_last_of("/");
        if (found != std::string::npos)
            name.erase(0, found + 1);
        name_test_images.push_back(name);
    }
    std::cout << "Test images loaded: " << test_images.size() << "/" << filenames.size() << "\n\n";
    return 0;
}

void Model::extractKeypointsFromViews()
{
    int th = 90;
    if (!datasetName.compare("driller"))
        th = 30;
    // 90 can/duck
    // 30 driller
    std::cout << "Extracting keypoints from views... ";

    for (int i = 0; i < views.size(); i++)
    {
        std::vector<descriptor> tmp_descr;
        getDiscriminativeLocations(views[i], tmp_descr, th);
        descriptors.push_back(tmp_descr);
    }
    std::cout << " Finished." << std::endl;
}

void Model::computePhaseImages()
{
    for (int i = 0; i < test_images.size(); i++)
    {
        cv::Mat output_magnitue, output_phase, output_phase_spread;
        computeGradientImage(test_images[i], output_magnitue, output_phase);
        output_phase_spread = cv::Mat(output_phase.rows, output_phase.cols, CV_8U, cv::Scalar(0));
        spreadPhase(output_phase, output_phase_spread, 3, true, false);
        phaseMaps.push_back(output_phase_spread);
    }
}

// compute one response map for each quantized phase
void Model::computeResponseMaps()
{
    // 8 response maps
    for (int i = 0; i < 8; i++)
    {
        // response map that correspond to current quantized phase i
        std::map<u_char, float> map;
        // quantized and encoded reference phase that correspond to the lookup table with index i
        int quant_ref_phase = 1 << i;
        // reference phase in degress
        float ref_phase_radians = reference_phases_rad[i];
        // key is a combination of phases encoded in a byte value
        // for example 01100000 -> represents the quantized phases 6 and 7
        // 255 possible encodings
        for (int key = 1; key < 256; key++)
        {
            float max = 0;
            // check which phases are contained in this key
            for (int quant_phase = 0; quant_phase < 8; quant_phase++)
            {
                float quant_phase_radians = reference_phases_rad[quant_phase];
                if ((u_char)key & (u_char)(1 << quant_phase))
                {
                    // key contains this quant_phase
                    float value = abs(cos(ref_phase_radians - quant_phase_radians));
                    if (value > max)
                        max = value;
                }
            }
            map.insert({(u_char)key, max});
        }
        response_maps.push_back(map);
    }
}

// compute scores at each location for each pair (image,view)
int Model::computeAllScores()
{
    std::cout << "Computing scores:\n";
    // for each test image
    for (int i = 0; i < test_images.size(); i++)
    {
        std::cout << "test_image: " << name_test_images[i] << std::endl;
        std::vector<score_info> res_i;
        //for each view
        for (int j = 0; j < views.size(); j++)
        {
            float max = 0;
            int x_max = 0, y_max = 0;
            // for each location in the test image
            for (int x = 0; x < test_images[i].cols; x++)
            {
                for (int y = 0; y < test_images[i].rows; y++)
                {

                    float tmp_score;
                    if ((tmp_score = computeScore(i, j, cv::Point(x, y))) < 0)
                    {
                        std::cout << "ERROR.";
                        return -1;
                    }
                    if (tmp_score > max)
                    {
                        max = tmp_score;
                        x_max = x;
                        y_max = y;
                    }
                }
            }
            res_i.push_back(score_info{j, max, x_max, y_max});
            std::cout << "  " << name_views[j] << "  score:" << max << " position:(" << x_max << "," << y_max << ")" << std::endl;
        }
        // sort list by decreasing score
        std::sort(res_i.begin(), res_i.end(), sortByScore);
        scores.push_back(res_i);
    }
    return 0;
}

// compute score at location c
float Model::computeScore(int index_image, int index_view, cv::Point c)
{
    float score = 0;
    std::vector<descriptor> set_locations = descriptors[index_view];

    for (int i = 0; i < set_locations.size(); i++)
    {
        cv::Point r = set_locations[i].loc;

        u_char phase_view = 0;
        u_char counter = set_locations[i].phase;

        if (!counter)
        {
            std::cout << "ERROR: phase must be positive\n";
            return -1;
        }

        while (counter != 1)
        {
            counter = counter >> 1;
            phase_view++;
        }

        // check that r+c is inside the image
        if ((r + c).x < test_images[index_image].cols && (r + c).y < test_images[index_image].rows)
        {
            // get phase at r+c on the test image
            u_char phase = phaseMaps[index_image].at<u_char>(r + c);
            //std::cout << "phase (r+c) =" << (int)phase << " phase template = " << (int)set_locations[i].phase << std::endl;
            if (phase)
            {
                score += response_maps[phase_view].at(phase);
            }
        }
    }
    return score;
}

void Model::initReferencePhases()
{
    // (180/8)/2 = 11.25 is the reference phase in radians for the first of the 8 sections
    float unit_phase = (11.25 / 180) * M_PI;
    for (int i = 0; i < 8; i++)
    {
        reference_phases_rad.push_back((2 * i + 1) * unit_phase);
    }
}

int Model::writeScores()
{
    ResultsWriter res_writer(datasetName, "..");
    for (int i = 0; i < scores.size(); i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (!res_writer.addResults(name_test_images[i], name_views[scores[i][j].index_view], scores[i][j].x, scores[i][j].y))
            {
                std::cout << "Ops! Something goes wrong..." << std::endl;
                return 1;
            }
        }
    }
    // Finally write to file
    if (!res_writer.write())
    {
        std::cout << "Ops! Something goes wrong..." << std::endl;
        return 1;
    }
    std::cout << "The scores have been written to file." << std::endl;
    return 0;
}

// External functions

// given a template compute a set of discriminative locations
void getDiscriminativeLocations(cv::Mat &view, std::vector<descriptor> &good_descriptors, int th)
{
    cv::Mat magnitude, phase;
    cv::Mat magnitude_refined, phase_refined;
    // 8 matrices, one for each phase encoding, and for each channel
    cv::Mat grad_x, grad_y;

    cv::Sobel(view, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(view, grad_y, CV_32F, 0, 1, 3);

    // compute magnitude and phase (in radians)
    cv::cartToPolar(grad_x, grad_y, magnitude, phase);
    //normalize and convert to CV_8UC3
    normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8UC3);

    computeMostFreqPhaseRGB(magnitude, phase, magnitude_refined, phase_refined, th);

    // compute a first list of candidate descriptors
    std::vector<descriptor> descriptors;
    for (int row = 0; row < magnitude_refined.rows; row++)
    {
        for (int col = 0; col < magnitude_refined.cols; col++)
        {
            if (magnitude_refined.at<u_char>(row, col) > th)
                descriptors.push_back(descriptor{cv::Point(col, row), magnitude_refined.at<u_char>(row, col), phase_refined.at<u_char>(row, col)});
        }
    }

    // sort the descriptors by decreasing magnitude
    std::sort(descriptors.begin(), descriptors.end(), sortByMagn);

    // compute the final list of descriptors, such that the distance of each pair of descriptors is greater than 'radius_dist'
    //std::cout << "Good descriptors:" << std::endl;
    good_descriptors.clear();
    cv::Mat maskDistance(view.rows, view.cols, CV_8U, cv::Scalar(0));
    cv::Mat debugImage = magnitude_refined.clone();
    int radius_dist = 7;
    bool start = true;
    // number of descriptors
    int n_descr = 40;
    if (descriptors.size() < n_descr)
        n_descr = descriptors.size();

    while (good_descriptors.size() != n_descr)
    {
        if (radius_dist < 0)
        {
            std::cout << "ERROR: less than" << n_descr << "descriptors found.\n";
            break;
        }
        if (start)
            start = false;
        else
        {
            radius_dist--;
            good_descriptors.clear();
            good_descriptors.shrink_to_fit();
            maskDistance = cv::Mat::zeros(view.rows, view.cols, CV_8U);
            debugImage = magnitude_refined.clone();
        }
        for (int i = 0; i < descriptors.size(); i++)
        {
            if (!maskDistance.at<u_char>(descriptors[i].loc))
            {
                good_descriptors.push_back(descriptors[i]);
                cv::circle(maskDistance, descriptors[i].loc, radius_dist, cv::Scalar(255), cv::FILLED);
                cv::drawMarker(debugImage, descriptors[i].loc, cv::Scalar(255), cv::MARKER_TRIANGLE_UP, 5);
            }
            if (good_descriptors.size() == n_descr)
                break;
        }
    }
}

//given an image, compute the magnitude and the phase matrix with the approach described in the paper
void computeGradientImage(cv::Mat &input_img, cv::Mat &output_magnitude, cv::Mat &output_phase)
{
    cv::Mat magnitude, phase;
    cv::Mat grad_x, grad_y;

    // gaussian blurr
    cv::Mat img_smoothed;
    cv::GaussianBlur(input_img, img_smoothed, cv::Size(3, 3), 3, 3);
    // compute gradient
    cv::Sobel(img_smoothed, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(img_smoothed, grad_y, CV_32F, 0, 1, 3);
    // compute magnitude and phase (in radians)
    cv::cartToPolar(grad_x, grad_y, magnitude, phase);

    //normalize and convert to CV_8UC3
    normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8UC3);

    computeMostFreqPhaseRGB(magnitude, phase, output_magnitude, output_phase, 10);
}

// given a magnitude image with 3 channel and a phase image with 3 channel
// compute the most frequent phase in a 3x3 patch, and select the strongest phase of the channel
void computeMostFreqPhaseRGB(cv::Mat &magnitude, cv::Mat &phase, cv::Mat &magnitude_refined, cv::Mat &phase_refined, int th)
{
    std::vector<cv::Mat> magnitude_BGR(3), phase_BGR(3), quant_phase_BGR(3);
    // 8 matrices, one for each phase encoding, and for each channel
    std::vector<cv::Mat> counter_phase_freq_B(8), counter_phase_freq_G(8), counter_phase_freq_R(8);

    // split gradient and phase channels
    cv::split(magnitude, magnitude_BGR);
    cv::split(phase, phase_BGR);

    // initialize phase (quantized) matrix for each channel
    quant_phase_BGR[0] = cv::Mat(magnitude_BGR[0].rows, magnitude_BGR[0].cols, CV_8U, cv::Scalar(0));
    quant_phase_BGR[1] = cv::Mat(magnitude_BGR[0].rows, magnitude_BGR[0].cols, CV_8U, cv::Scalar(0));
    quant_phase_BGR[2] = cv::Mat(magnitude_BGR[0].rows, magnitude_BGR[0].cols, CV_8U, cv::Scalar(0));

    // initialize phase counters
    for (int i = 0; i < 8; i++)
    {
        counter_phase_freq_B[i] = cv::Mat(magnitude_BGR[0].rows, magnitude_BGR[0].cols, CV_8U, cv::Scalar(0));
        counter_phase_freq_G[i] = cv::Mat(magnitude_BGR[0].rows, magnitude_BGR[0].cols, CV_8U, cv::Scalar(0));
        counter_phase_freq_R[i] = cv::Mat(magnitude_BGR[0].rows, magnitude_BGR[0].cols, CV_8U, cv::Scalar(0));
    }

    //quantize phase & select the most frequent phase that occurs in a patch 3x3
    for (int r = 0; r < magnitude_BGR[0].rows; r++)
    {
        for (int c = 0; c < magnitude_BGR[0].cols; c++)
        {
            // blue channel
            if (magnitude_BGR[0].at<u_char>(r, c) > th)
            {
                quant_phase_BGR[0].at<u_char>(r, c) = quantizePhase(phase_BGR[0].at<float>(r, c));
                increaseCounterPhase(counter_phase_freq_B, r, c, quant_phase_BGR[0].at<u_char>(r, c), 3, false);
            }
            // green channel
            if (magnitude_BGR[1].at<u_char>(r, c) > th)
            {
                quant_phase_BGR[1].at<u_char>(r, c) = quantizePhase(phase_BGR[1].at<float>(r, c));
                increaseCounterPhase(counter_phase_freq_G, r, c, quant_phase_BGR[1].at<u_char>(r, c), 3, false);
            }
            // red channel
            if (magnitude_BGR[2].at<u_char>(r, c) > th)
            {
                quant_phase_BGR[2].at<u_char>(r, c) = quantizePhase(phase_BGR[2].at<float>(r, c));
                increaseCounterPhase(counter_phase_freq_R, r, c, quant_phase_BGR[2].at<u_char>(r, c), 3, false);
            }
        }
    }

    // for each channel compute the most frequent phase for each location in a 3x3 patch
    computeMostFreqPhase(counter_phase_freq_B, quant_phase_BGR[0]);
    computeMostFreqPhase(counter_phase_freq_G, quant_phase_BGR[1]);
    computeMostFreqPhase(counter_phase_freq_R, quant_phase_BGR[2]);

    // select for each location the phase with the strongest magnitude
    selectStrongestPhase(magnitude_BGR, quant_phase_BGR, magnitude_refined, phase_refined, false);
}

// for each location, select the phase with the largest magnitude
// flag quantize true = phase to be quantized (false = phase already quantized)
void selectStrongestPhase(std::vector<cv::Mat> &magnitude_BGR, std::vector<cv::Mat> &phase_BGR, cv::Mat &output_magnitude, cv::Mat &output_phase, bool quantize)
{
    // sanity check
    if (magnitude_BGR[0].size() != magnitude_BGR[1].size() || magnitude_BGR[1].size() != magnitude_BGR[2].size())
    {
        std::cerr << "ERROR: Input matrices with different size." << std::endl;
    }

    int rows = magnitude_BGR[0].rows;
    int cols = magnitude_BGR[0].cols;
    output_magnitude = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
    output_phase = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int max = magnitude_BGR[0].at<u_char>(r, c);
            int phase = phase_BGR[0].at<u_char>(r, c);
            if (magnitude_BGR[1].at<u_char>(r, c) > max)
            {
                max = magnitude_BGR[1].at<u_char>(r, c);
                phase = phase_BGR[1].at<u_char>(r, c);
            }
            if (magnitude_BGR[2].at<u_char>(r, c) > max)
            {
                max = magnitude_BGR[2].at<u_char>(r, c);
                phase = phase_BGR[2].at<u_char>(r, c);
            }
            if (quantize)
                phase = quantizePhase(phase);
            output_phase.at<u_char>(r, c) = phase;
            output_magnitude.at<u_char>(r, c) = max;
        }
    }
}

// flag quantize true = phase to be quantized (false = phase already quantized)
// bool binary_count true -> each cell can be 0/1
// bool binary_count false -> int values
void spreadPhase(cv::Mat &input_phase, cv::Mat &output_phase_spread, int wind_size, bool binary_count, bool quantize)
{
    // one matrix for each quantized phase
    std::vector<cv::Mat> binary_phases(8);
    output_phase_spread = cv::Mat(input_phase.rows, input_phase.cols, CV_8U, cv::Scalar(0));
    // initialize binary phase matrices
    for (int i = 0; i < 8; i++)
    {
        binary_phases[i] = cv::Mat(input_phase.rows, input_phase.cols, CV_8U, cv::Scalar(0));
    }
    // shift each phase in the corresponding phase matrix
    for (int r = 0; r < input_phase.rows; r++)
    {
        for (int c = 0; c < input_phase.cols; c++)
        {
            u_char phase = input_phase.at<u_char>(r, c);
            if (quantize)
                phase = quantizePhase(phase);
            increaseCounterPhase(binary_phases, r, c, phase, wind_size, binary_count);
        }
    }

    // encode correctly the phase values
    for (int i = 0; i < 8; i++)
    {
        binary_phases[i] *= pow(2, i);
    }
    //std::cout << "binary_phases:\n" << binary_phases[2] << std::endl;

    // merge shifted phase matrices
    for (int i = 0; i < 8; i++)
    {
        output_phase_spread = output_phase_spread | binary_phases[i];
    }
    //std::cout << "output_phase_spread:\n" << output_phase_spread << std::endl;
}

// increase the counter of the phase on window of size wind_size
// bool binary_count true -> each cell can be 0/1
// bool binary_count false -> int values
void increaseCounterPhase(std::vector<cv::Mat> &vector_counters, int row, int col, unsigned char phase, int wind_size, bool binary_count)
{
    int max_row = vector_counters[0].rows - 1; //index
    int max_col = vector_counters[0].cols - 1; //index
    // sanity check
    if (row > max_row || col > max_col)
    {
        std::cerr << "ERROR: row or col too large." << std::endl;
    }
    if (row < 0 || col < 0)
    {
        std::cerr << "ERROR: row and col must be positive values." << std::endl;
    }
    // if phase = 00000000 (it does not encode any phase)
    if (!phase)
        return;
    // index to access the matrix that correspond to the phase passed as argument
    int index_phase = log2(phase); //index
    int offset = (wind_size - 1) / 2;
    int first_row = std::max(row - offset, 0);
    int last_row = std::min(row + offset, max_row);
    int first_col = std::max(col - offset, 0);
    int last_col = std::min(col + offset, max_col);

    for (int r = first_row; r <= last_row; r++)
    {
        for (int c = first_col; c <= last_col; c++)
        {
            if (binary_count)
            {
                vector_counters[index_phase].at<u_char>(r, c) = 1;
            }
            else
            {
                vector_counters[index_phase].at<u_char>(r, c)++;
            }
        }
    }
}

// compute most frequent quantized phase in a patch of 3x3 for every location
void computeMostFreqPhase(std::vector<cv::Mat> &vector_counters, cv::Mat &output_phase)
{
    int n_phase = vector_counters.size();
    int rows = vector_counters[0].rows;
    int cols = vector_counters[0].cols;
    output_phase = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int max = 0;
            unsigned phase = 0; // null phase encoding
            for (int i = 0; i < n_phase; i++)
            {
                if ((int)vector_counters[i].at<u_char>(r, c) > max)
                {
                    max = vector_counters[i].at<u_char>(r, c);
                    phase = 1 << i;
                }
            }
            output_phase.at<u_char>(r, c) = phase;
        }
    }
}

// quantize phase
unsigned char quantizePhase(float phase)
{
    float tmp = phase;
    // [0:pi)
    if (tmp >= M_PI)
        tmp -= M_PI;
    int shift = (int)((tmp * 8) / M_PI);
    return 1 << shift;
}

// utility comparator function to pass to the sort() module
bool sortByMagn(const descriptor &a,
                const descriptor &b)
{
    return (a.magn > b.magn);
}
// utility comparator function to pass to the sort() module
bool sortByScore(const score_info &a,
                 const score_info &b)
{
    return (a.max_score > b.max_score);
}
