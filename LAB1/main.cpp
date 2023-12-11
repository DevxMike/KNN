#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_map>

struct record_t {
    using record_type = std::vector<double>;

    record_type record;
    int gesture_id;
};



using matrix_type = std::vector<record_t>;

std::ostream& operator<<(std::ostream& os, const record_t& r) {
    os << r.gesture_id << ": ";

    for (const auto& x : r.record) {
        os << x << " ";
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const matrix_type& m) {
    for (const auto& x : m) {
        os << x << std::endl;
    }

    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& confusion_matrix) {
    for (const auto& confusion_record : confusion_matrix) {
        for (const auto& x : confusion_record) {
            os << x << " ";
        }
        os << '\n';
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::pair<int, double>>& acc) {
    for (const auto& p : acc) {
        os << p.first << " - " << p.second << " ";
    }
    os << std::endl;

    return os;
}

void load_data(matrix_type& m, std::string fname) {
    try {
        std::ifstream fin(fname);

        if (!fin.is_open()) {
            throw;
        }

        std::string line;

        while (std::getline(fin, line)) {
            std::istringstream iss(line);

            record_t new_record;

            iss >> new_record.gesture_id;

            new_record.gesture_id--;

            double value;
            while (iss >> value) {
                new_record.record.push_back(value);
            }
            m.push_back(new_record);
        }
    }
    catch (...) {
        exit(-1);
    }
}

double euclidean_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double dist = 0.0;

    for (int i = 0; i < a.size(); ++i) {
        dist += std::pow(a[i] - b[i], 2);
    }

    return std::sqrt(dist);
}

double manhattan_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double dist = 0.0;

    for (int i = 0; i < a.size(); ++i) {
        dist += std::abs(a[i] - b[i]);
    }

    return std::sqrt(dist);
}

using dist_function = std::function<double(const std::vector<double>&, const std::vector<double>&)>;

int knn_predict(const matrix_type& m, const std::vector<double>& features, int k, dist_function get_dist) {
    std::vector<std::pair<double, int>> distances_and_labels;

    for (const auto& record : m) {
        distances_and_labels.emplace_back(
            get_dist(record.record, features),
            record.gesture_id
        );
    }

    std::sort(distances_and_labels.begin(), distances_and_labels.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        }
    );

    std::unordered_map<int, int> class_votes;

    for (int i = 0; i < k; ++i) {
        class_votes[distances_and_labels[i].second]++;
    }

    auto predicted = std::max_element(class_votes.begin(), class_votes.end(),
        [](const auto& e1, const auto& e2) {
            return e1.second < e2.second;
        }
    );

    if (predicted != class_votes.end()) {
        return predicted->first;
    }

    return -1;
}

std::vector<std::vector<int>> get_confusion_matrix_count(const matrix_type& training_data, const matrix_type& testing_data, int k, dist_function get_dist) {
    auto tmp = std::max_element(training_data.begin(), training_data.end(),
        [](const auto& r1, const auto& r2) {
            return r1.gesture_id < r2.gesture_id;
        }
    );

    if (tmp == training_data.end()) {
        throw(std::out_of_range("Error"));
    }

    auto num_classes = tmp->gesture_id + 1;

    std::vector<std::vector<int>> prediction_matrix(num_classes, std::vector<int>(num_classes, 0));

    for (const auto& test_record : testing_data) {
        auto predicted = knn_predict(training_data, test_record.record, k, get_dist);
        prediction_matrix[test_record.gesture_id][predicted]++;
    }

    return prediction_matrix;
}

std::vector<std::vector<double>> get_confusion_matrix_percentage(const matrix_type& training_data, const matrix_type& testing_data, int k, dist_function get_dist) {
    auto tmp = std::max_element(training_data.begin(), training_data.end(),
        [](const auto& r1, const auto& r2) {
            return r1.gesture_id < r2.gesture_id;
        }
    );

    if (tmp == training_data.end()) {
        throw(std::out_of_range("Error"));
    }

    auto num_classes = tmp->gesture_id + 1;
    
    std::vector<std::vector<int>> prediction_matrix = get_confusion_matrix_count(training_data, testing_data, k, get_dist);
    std::vector<std::vector<double>> result(num_classes, std::vector<double>());

    int i = 0;

    for (auto& prediction_record : prediction_matrix) {
        auto prediction_count = std::accumulate(prediction_record.begin(), prediction_record.end(), 0);

        for (auto& x : prediction_record) {
            result[i].push_back((1.0 * x / prediction_count) * 100.0);
        }
        ++i;
    }

    return result;
}

double get_acc(const matrix_type& training_data, const matrix_type& testing_data, int k, dist_function get_dist) {
    int total = 0, correct = 0;

    for (const auto& test_record : testing_data) {
        auto prediction = knn_predict(training_data, test_record.record, k, get_dist);

        if (prediction == test_record.gesture_id) {
            ++correct;
        }

        ++total;
    }

    return (1.0 * correct / total) * 100.0;
}

int main(void) {
    matrix_type training_data, testing_data;

    load_data(training_data, "training.dat");
    load_data(testing_data, "testing.dat");

    int total = 0;
    int correct = 0;

    for (const auto& test_record : testing_data) {
        auto tmp = knn_predict(training_data, test_record.record, 30, euclidean_dist);

        if (tmp == test_record.gesture_id) {
            correct++;
        }

        total++;
    }

    auto euclidean_percentage = 100.0 * correct / total;

    std::cout << "Euclidean: " << euclidean_percentage << '%' << std::endl;

    total = correct = 0;

    for (const auto& test_record : testing_data) {
        auto tmp = knn_predict(training_data, test_record.record, 30, manhattan_dist);

        if (tmp == test_record.gesture_id) {
            correct++;
        }

        total++;
    }

    auto manhattan_percentage = 100.0 * correct / total;

    std::cout << "Manhattan: " << manhattan_percentage << '%' << std::endl;

    auto result_euclidean_percentage = get_confusion_matrix_percentage(training_data, testing_data, 10, euclidean_dist);
    auto result_manhattan_percentage = get_confusion_matrix_percentage(training_data, testing_data, 10, manhattan_dist);

    std::cout << "Confusion Matrix euclidean (percentage): " << std::endl << result_euclidean_percentage << std::endl;

    std::cout << "Confusion Matrix manhattan (percentage): " << std::endl << result_manhattan_percentage << std::endl;

    auto result_euclidean_count = get_confusion_matrix_count(training_data, testing_data, 10, euclidean_dist);
    auto result_manhattan_count = get_confusion_matrix_count(training_data, testing_data, 10, manhattan_dist);

    std::cout << "Confusion Matrix euclidean (count): " << std::endl << result_euclidean_count << std::endl;

    std::cout << "Confusion Matrix manhattan (count): " << std::endl << result_manhattan_count << std::endl;

    std::ofstream fout("result.txt", std::ios_base::out);

    try {
        fout << "Euclidean: " << euclidean_percentage << '%' << std::endl;
        fout << "Manhattan: " << manhattan_percentage << '%' << std::endl;
        fout << "Confusion Matrix euclidean (percentage): " << std::endl << result_euclidean_percentage << std::endl;
        fout << "Confusion Matrix manhattan (percentage): " << std::endl << result_manhattan_percentage << std::endl;
        fout << "Confusion Matrix euclidean (count): " << std::endl << result_euclidean_count << std::endl;
        fout << "Confusion Matrix manhattan (count): " << std::endl << result_manhattan_count << std::endl;
        fout.close();
    }
    catch (...) {
        std::cout << "Error occured while opening a file!\n";

        return -1;
    }

    return 0;
}