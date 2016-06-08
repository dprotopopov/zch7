// Алгоритм многомерной оптимизации с использованием метода решёток

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric>
#include <locale>
#include <assert.h>
#include <fstream>

using namespace std;


double module(std::vector<double>& x);
double delta(std::vector<double>& x, std::vector<double>& y);
unsigned long total_of(std::vector<size_t>& m);
void vector_of(std::vector<unsigned>& vector, unsigned long index, std::vector<size_t>& m);
void point_of(std::vector<double>& point, std::vector<unsigned>& vector, std::vector<size_t>& m, std::vector<double>& a, std::vector<double>& b);

template <typename T>
T inc_functor(T value)
{
	return ++value;
}

template <typename T>
T square_functor(T value)
{
	return value * value;
}

template <typename T>
T add_functor(T value1, T value2)
{
	return value1 + value2;
}

template <typename T>
T sub_functor(T value1, T value2)
{
	return value1 - value2;
}

template <typename T>
T mul_functor(T value1, T value2)
{
	return value1 * value2;
}

template <typename T>
T abs_functor(T value)
{
	return std::abs(value);
}

template <typename T>
T diff_functor(T value1, T value2)
{
	return std::abs(value1 - value2);
}

template <typename T>
T max_functor(T value1, T value2)
{
	return std::max(value1, value2);
}

bool and_functor(const bool value1, const bool value2)
{
	return value1 && value2;
}

bool or_functor(const bool value1, const bool value2)
{
	return value1 || value2;
}

enum t_ask_mode
{
	NOASK = 0,
	ASK = 1
};

enum t_trace_mode
{
	NOTRACE = 0,
	TRACE = 1
};

t_ask_mode ask_mode = NOASK;
t_trace_mode trace_mode = NOTRACE;

/////////////////////////////////////////////////////////
// Дефолтные значения
static const unsigned _count = 1;
static const size_t _n = 4;
static const size_t _m[] = {10, 10, 10, 10};
static const double _a[] = {0, 0, 0, 0};
static const double _b[] = {1000, 1000, 1000, 1000};
static const double _e = 1e-8;

typedef struct history_t {
	double a;
	double b;
	double c;
	double d;
	double y;
} history_t;

/////////////////////////////////////////////////////////
// Вычисление модуля вектора
double module(std::vector<double>& x)
{
	std::vector<double> y(x.size());
	std::transform(x.begin(), x.end(), y.begin(), abs_functor<double>);
	return std::accumulate(y.begin(), y.end(), 0.0, max_functor<double>);
}

/////////////////////////////////////////////////////////
// Вычисление растояния между двумя векторами координат
double delta(std::vector<double>& x, std::vector<double>& y)
{
	size_t i = std::min(x.size(), y.size());
	std::vector<double> diff(std::max(x.size(), y.size()));
	std::transform(x.begin(), x.begin() + i, y.begin(), diff.begin(), diff_functor<double>);
	std::transform(x.begin() + i, x.end(), diff.begin() + i, abs_functor<double>);
	std::transform(y.begin() + i, y.end(), diff.begin() + i, abs_functor<double>);
	return std::accumulate(diff.begin(), diff.end(), 0.0, max_functor<double>);
}

/////////////////////////////////////////////////////////
// Вычисление вектора индексов координат решётки по номеру узла
// vector - вектор индексов координат решётки
// index - номер узла решётки
// m - число сегментов по каждому из измерений
void vector_of(std::vector<unsigned>& vector, unsigned long index, std::vector<size_t>& m)
{
	for (size_t i = 0; i < m.size(); i++)
	{
		unsigned long m1 = 1ul + m[i];
		vector[i] = index % m1;
		index /= m1;
	}
}

/////////////////////////////////////////////////////////
// Преобразование вектора индексов координат решётки
// в вектор координат точки
// vector - вектор индексов координат решётки
// m - число сегментов по каждому из измерений
// a - вектор минимальных координат точек
// b - вектор максимальных координат точек
void point_of(std::vector<double>& point, std::vector<unsigned>& vector, std::vector<size_t>& m, std::vector<double>& a, std::vector<double>& b)
{
	for (size_t i = 0; i < m.size(); i++) point[i] = (a[i] * (m[i] - vector[i]) + b[i] * vector[i]) / m[i];
}

/////////////////////////////////////////////////////////
// Вычисление числа узлов решётки
// m - число сегментов по каждому из измерений
unsigned long total_of(std::vector<size_t>& m)
{
	std::vector<size_t> m1(m.size());
	std::transform(m.begin(), m.end(), m1.begin(), inc_functor<size_t>);
	return std::accumulate(m1.begin(), m1.end(), 1UL, mul_functor<unsigned long>);
}

/////////////////////////////////////////////////////////
// Искомая функция
double target(std::vector<double>& x, std::vector<history_t>& history)
{
	size_t n = x.size();
	assert(n == 4);
	std::vector<double> s(history.size());
#pragma omp parallel for
	for (size_t i = 0; i < s.size(); i++)
	{
		history_t &h = history[i];
		double y1 = h.a*x[0] + h.a*h.b*(x[1] + h.c*x[2] + h.d*x[3]);
		s[i] = (h.y - y1)*(h.y - y1);
	}
	return std::accumulate(s.begin(), s.end(), 0.0, add_functor<double>);
}


int main(int argc, char* argv[])
{
	// http://stackoverflow.com/questions/2236197/what-is-the-easiest-way-to-initialize-a-stdvector-with-hardcoded-elements

	unsigned count = _count;
	size_t n = _n;
	double e = _e;
	std::vector<size_t> m(_m, _m + sizeof(_m) / sizeof(_m[0]));
	std::vector<double> a(_a, _a + sizeof(_a) / sizeof(_a[0]));
	std::vector<double> b(_b, _b + sizeof(_b) / sizeof(_b[0]));
	std::vector<history_t> history;

	char* input_file_name = NULL;
	char* output_file_name = NULL;
	char* options_file_name = NULL;

	// Поддержка кириллицы в консоли Windows
	// Функция setlocale() имеет два параметра, первый параметр - тип категории локали, в нашем случае LC_TYPE - набор символов, второй параметр — значение локали. 
	// Вместо второго аргумента можно писать "Russian", или оставлять пустые двойные кавычки, тогда набор символов будет такой же как и в ОС.
	setlocale(LC_ALL, "");
	setlocale(LC_NUMERIC, "C");

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-help") == 0)
		{
			std::cout << "Usage :\t" << argv[0] << " [...] [-input <inputfile>] [-output <outputfile>]" << std::endl;
			std::cout << "Алгоритм многомерной оптимизации с использованием метода решёток" << std::endl;
			std::cout << "Алгоритм деления значений аргумента функции" << std::endl;
			//			std::cout << "\t-n <размерность пространства>" << std::endl;
			std::cout << "\t-c <количество повторений алгоритма для замера времени>" << std::endl;
			std::cout << "\t-m <число сегментов по каждому из измерений>" << std::endl;
			std::cout << "\t-a <минимальные координаты по каждому из измерений>" << std::endl;
			std::cout << "\t-b <максимальные координаты по каждому из измерений>" << std::endl;
			std::cout << "\t-e <точность вычислений>" << std::endl;
			std::cout << "\t-ask/noask" << std::endl;
			std::cout << "\t-trace/notrace" << std::endl;
		}
		else if (strcmp(argv[i], "-ask") == 0) ask_mode = ASK;
		else if (strcmp(argv[i], "-noask") == 0) ask_mode = NOASK;
		else if (strcmp(argv[i], "-trace") == 0) trace_mode = TRACE;
		else if (strcmp(argv[i], "-notrace") == 0) trace_mode = NOTRACE;
		else if (strcmp(argv[i], "-n") == 0) n = atoi(argv[++i]);
		else if (strcmp(argv[i], "-e") == 0) e = atof(argv[++i]);
		else if (strcmp(argv[i], "-c") == 0) count = atoi(argv[++i]);
		else if (strcmp(argv[i], "-m") == 0)
		{
			std::istringstream ss(argv[++i]);
			m.clear();
			for (size_t i1 = 0; i1 < n; i1++) m.push_back(atoi(argv[++i]));
		}
		else if (strcmp(argv[i], "-a") == 0)
		{
			a.clear();
			for (size_t i1 = 0; i1 < n; i1++) a.push_back(atof(argv[++i]));
		}
		else if (strcmp(argv[i], "-b") == 0)
		{
			b.clear();
			for (size_t i1 = 0; i1 < n; i1++) b.push_back(atof(argv[++i]));
		}
		else if (strcmp(argv[i], "-input") == 0) input_file_name = argv[++i];
		else if (strcmp(argv[i], "-output") == 0) output_file_name = argv[++i];
		else if (strcmp(argv[i], "-options") == 0) options_file_name = argv[++i];
	}

	if (input_file_name != NULL) freopen(input_file_name, "r",stdin);
	if (output_file_name != NULL) freopen(output_file_name, "w",stdout);

	if (options_file_name != NULL)
	{
		std::ifstream options(options_file_name);
		if (!options.is_open()) throw "Error opening file";
		std::string line;
		while (std::getline(options, line))
		{
			std::cout << line << std::endl;
			std::stringstream lineStream(line);
			std::string id;
			std::string cell;
			std::vector<double> x;
			std::vector<size_t> y;
			std::getline(lineStream, id, ' ');
			if (id[0] == 'E')
			{
				std::getline(lineStream, cell, ' ');
				e = stod(cell);
			}
			if (id[0] == 'M')
			{
				while (std::getline(lineStream, cell, ' '))
				{
					y.push_back(stoi(cell));
				}
				m = y;
			}
			if (id[0] == 'A')
			{
				while (std::getline(lineStream, cell, ' '))
				{
					x.push_back(stod(cell));
				}
				a = x;
			}
			if (id[0] == 'B')
			{
				while (std::getline(lineStream, cell, ' '))
				{
					x.push_back(stod(cell));
				}
				b = x;
			}
		}
	}

	if (ask_mode == ASK)
	{
		//std::cout << "Введите размерность пространства:" << std::endl;
		//std::cin >> n;

		std::cout << "Введите число сегментов по каждому из измерений m[" << n << "]:" << std::endl;
		m.clear();
		for (size_t i = 0; i < n; i++)
		{
			int x;
			std::cin >> x;
			m.push_back(x);
		}

		std::cout << "Введите минимальные координаты по каждому из измерений a[" << n << "]:" << std::endl;
		a.clear();
		for (size_t i = 0; i < n; i++)
		{
			double x;
			std::cin >> x;
			a.push_back(x);
		}

		std::cout << "Введите максимальные координаты по каждому из измерений b[" << n << "]:" << std::endl;
		b.clear();
		for (size_t i = 0; i < n; i++)
		{
			double x;
			std::cin >> x;
			b.push_back(x);
		}

		std::cout << "Введите точность вычислений:" << std::endl;
		std::cin >> e;
		std::cout << "Введите количество повторений алгоритма для замера времени:" << std::endl;
		std::cin >> count;
	}

	for (size_t i = 0; i < m.size(); i++) assert(m[i]>2);

	std::string line;
	while (std::getline(std::cin, line))
	{
		std::cout << line << std::endl;
		std::stringstream lineStream(line);
		history_t h;
		lineStream >> h.a >> h.b >> h.c >> h.d >> h.y;
		history.push_back(h);
	}

	// Алгоритм
	clock_t time = clock();

	std::vector<unsigned> v(n);
	std::vector<double> x(n);
	std::vector<double> a1(n);
	std::vector<double> b1(n);
	double y;

	if (trace_mode == TRACE && count == 1) std::cout << "for #1" << std::endl;
	for (unsigned s = 0; s < count; s++)
	{
		std::copy(a.begin(), a.end(), a1.begin());
		std::copy(b.begin(), b.end(), b1.begin());

		if (trace_mode == TRACE && count == 1) std::cout << "while #1" << std::endl;
		while (true)
		{
			unsigned long total = total_of(m);
			int root = sqrt(total);
			unsigned long index = 0;

			vector_of(v, index, m);
			point_of(x, v, m, a1, b1);
			y = target(x, history);

			// Находим следующую точку в области, заданной ограничениями
			for (unsigned long index1 = index + 1; index1 < total; index1 += root)
			{
				std::vector<double> doubles(root, DBL_MAX);
#pragma omp parallel for
				for (int i = 0; i < root; i++)
					if (index1 + i < total)
					{
						std::vector<unsigned> v1(n);
						std::vector<double> t(n);
						vector_of(v1, index1 + i, m);
						point_of(t, v1, m, a1, b1);
						doubles[i] = target(t, history);
					}
				for (auto it = doubles.begin();it < doubles.end();++it)
				{
					if (*it > y) continue;
					y = *it;
					index = index1 + std::distance(doubles.begin(), it);
				}
			}
			vector_of(v, index, m);
			point_of(x, v, m, a1, b1);

			if (trace_mode == TRACE && count == 1) for (size_t i = 0; i < x.size(); i++) std::cout << x[i] << " ";
			if (trace_mode == TRACE && count == 1) std::cout << "-> " << y << std::endl;

			double dd = delta(a1, b1);
			double cc = std::max(module(a1), module(b1));
			if (dd <= cc * e) break;

#pragma omp parallel for
			for (size_t k = 0; k < n; k++)
			{
				double ak = a1[k];
				double bk = b1[k];
				double xk = x[k];
				double dd = std::max(ak - bk, bk - ak);
				a1[k] = std::max(ak, xk - dd / m[k]);
				b1[k] = std::min(bk, xk + dd / m[k]);
			}
		}
	}

	time = clock() - time;
	double seconds = ((double)time) / CLOCKS_PER_SEC / count;

	std::cout << "Исполняемый файл         : " << argv[0] << std::endl;
	std::cout << "Размерность пространства : " << n << std::endl;
	std::cout << "Число сегментов          : ";
	for (size_t i = 0; i < m.size(); i++) std::cout << m[i] << " ";
	std::cout << std::endl;
	std::cout << "Минимальные координаты   : ";
	for (size_t i = 0; i < a.size(); i++) std::cout << a[i] << " ";
	std::cout << std::endl;
	std::cout << "Максимальные координаты  : ";
	for (size_t i = 0; i < b.size(); i++) std::cout << b[i] << " ";
	std::cout << std::endl;
	std::cout << "Точность вычислений      : " << e << std::endl;
	std::cout << "Точка минимума           : ";
	for (size_t i = 0; i < x.size(); i++) std::cout << x[i] << " ";
	std::cout << std::endl;
	std::cout << "Минимальное значение     : " << y << std::endl;
	std::cout << "Время вычислений (сек.)  : " << seconds << std::endl;

	getchar();
	getchar();

	return 0;
}
