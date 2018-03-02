#include <string>
#include <iostream>
#include <fftw3.h>
#include <random>
#include <complex>
#include <cstring>
#include <string.h>
#include <memory.h>
#include <cmath>
#include <vector>
#include <float.h>

using namespace std;

int N = 1024;
int BINSIZE = 4;
int SCALE = 4;


////////////////////////////// RunningStats code ///////////////////////////
//////// from https://www.johndcook.com/blog/skewness_kurtosis/ ///////////
 
class RunningStats
{
public:
    RunningStats();
    void Clear();
    void Push(double x);
    long long NumDataValues() const;
    double Mean() const;
    double Variance() const;
    double StandardDeviation() const;
    double Skewness() const;
    double Kurtosis() const;
 
    friend RunningStats operator+(const RunningStats a, const RunningStats b);
    RunningStats& operator+=(const RunningStats &rhs);
 
private:
    long long n;
    double M1, M2, M3, M4;
};
 
// And here is the implementation file RunningStats.cpp.
 
RunningStats::RunningStats() 
{
    Clear();
}
 
void RunningStats::Clear()
{
    n = 0;
    M1 = M2 = M3 = M4 = 0.0;
}
 
void RunningStats::Push(double x)
{
    double delta, delta_n, delta_n2, term1;
 
    long long n1 = n;
    n++;
    delta = x - M1;
    delta_n = delta / n;
    delta_n2 = delta_n * delta_n;
    term1 = delta * delta_n * n1;
    M1 += delta_n;
    M4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
    M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
    M2 += term1;
}
 
long long RunningStats::NumDataValues() const
{
    return n;
}
 
double RunningStats::Mean() const
{
    return M1;
}
 
double RunningStats::Variance() const
{
    return M2/(n-1.0);
}
 
double RunningStats::StandardDeviation() const
{
    return sqrt( Variance() );
}
 
double RunningStats::Skewness() const
{
    return sqrt(double(n)) * M3/ pow(M2, 1.5);
}
 
double RunningStats::Kurtosis() const
{
    return double(n)*M4 / (M2*M2) - 3.0;
}
 
RunningStats operator+(const RunningStats a, const RunningStats b)
{
    RunningStats combined;
     
    combined.n = a.n + b.n;
     
    double delta = b.M1 - a.M1;
    double delta2 = delta*delta;
    double delta3 = delta*delta2;
    double delta4 = delta2*delta2;
     
    combined.M1 = (a.n*a.M1 + b.n*b.M1) / combined.n;
     
    combined.M2 = a.M2 + b.M2 + 
                  delta2 * a.n * b.n / combined.n;
     
    combined.M3 = a.M3 + b.M3 + 
                  delta3 * a.n * b.n * (a.n - b.n)/(combined.n*combined.n);
    combined.M3 += 3.0*delta * (a.n*b.M2 - b.n*a.M2) / combined.n;
     
    combined.M4 = a.M4 + b.M4 + delta4*a.n*b.n * (a.n*a.n - a.n*b.n + b.n*b.n) / 
                  (combined.n*combined.n*combined.n);
    combined.M4 += 6.0*delta2 * (a.n*a.n*b.M2 + b.n*b.n*a.M2)/(combined.n*combined.n) + 
                  4.0*delta*(a.n*b.M3 - b.n*a.M3) / combined.n;
     
    return combined;
}
 
RunningStats& RunningStats::operator+=(const RunningStats& rhs)
{ 
        RunningStats combined = *this + rhs;
        *this = combined;
        return *this;
}

///////////////////////// End of RunningStats code //////////////////////////


// Prints a distribution from a histogram array
// scale changes number of stars per bin
// binSize changes range of each bin 
void printDistribution(int hist[], int binSize = 1, int scale = 1){
	for(int i=0;i<255;i+=binSize){
        int stars = 0;
		for(int j=i;j<i+binSize;++j){
			stars+= hist[j];
		}
        for(int j=0;j<stars/scale;++j){
            cout << "*";
        }
        cout << endl;
    }
}

// Performs a 1D complex-to-complex FFT
void FFT(fftw_complex in[], fftw_complex out[]){
	fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
}

// Performs a 1D complex-to-complex IFFT
void IFFT(fftw_complex in[], fftw_complex out[]){
	fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
}

int main()
{
	// distribution generator
    mt19937 gen;
	// distribution
    normal_distribution<double> distr(0, 20);
	
	// setting up arrays to hold initial data and histograms
    char dataIn[N][4] = {0};
	int hist[4][255] = {0};
	
	// loop through data array
	for(int i=0;i<N;++i){
		for(int j=0;j<4;++j){
			char num = (char)(distr(gen)); // randomly generate character
			dataIn[i][j] = num; // record in data
			hist[j][num+128]++; // add to histogram
		}
	}
	
	// Histogram titles
	string words[4] = {"X Real", "X Imaginary", "Y Real", "Y Imaginary"};
	cout << endl;
	
	// Print all histograms
	for(int i=0;i<4;++i){
		cout << words[i] << endl << endl;
		printDistribution(hist[i], BINSIZE, SCALE);
	}
	
	// complex arrays to hold initial data
	fftw_complex xVals[N];
	fftw_complex yVals[N];
	
	// get initial data in fftw_complex format
	for(int i=0;i<N;++i){
		complex<double> x(dataIn[i][0], dataIn[i][1]);
		complex<double> y(dataIn[i][2], dataIn[i][3]);
		memcpy(&xVals[i], &x, sizeof(fftw_complex));
		memcpy(&yVals[i], &y, sizeof(fftw_complex));
	}
	
	// complex arrays to hold fourier transformed data
	fftw_complex xFFT[N];
	fftw_complex yFFT[N];
	
	// complex arrays to hold IFFT output data
	fftw_complex xOut[N];
    fftw_complex yOut[N];
	
	// Perform FFT/IFFT on initial data
	FFT(xVals, xFFT);
	FFT(yVals, yFFT);

	// Power spectra
	double xPow[N] = {0};
	double yPow[N] = {0};
	
	for(int i=0;i<N;++i){
		xPow[i] = pow(xFFT[i][0], 2)+pow(xFFT[i][1], 2);
		yPow[i] = pow(yFFT[i][0], 2)+pow(yFFT[i][1], 2);
	}

	
/////////// Kurtosis code goes here:

	RunningStats rs[6] = {RunningStats()};

	for(int i=0;i<N;++i){
		// put values in a temporary array for loop
		double d[] = {xFFT[i][0], xFFT[i][1], yFFT[i][0], yFFT[i][1], xPow[i], yPow[i]};
		for(int j=0;j<6;++j){
			rs[j].Push(d[j]);
		}
	}
	
	
		
	for(int i=0; i<6; ++i){
		cout << "Kurtosis for row " << i << ": " << rs[i].Kurtosis() << endl;
		cout << "N: " << rs[i].NumDataValues() << endl;
		cout << "Mean: " << rs[i].Mean() << endl;
		cout << "Variance: " << rs[i].Variance() << endl;
		cout << "Skewness: " << rs[i].Skewness() << endl << endl;
	}
	
////////// End of Kurtosis code	
	
	IFFT(xFFT, xOut);
	IFFT(yFFT, yOut);
	
	// array to hold char format of output data
	char dataOut[N][4] = {0};
//	double errors[N][4] = {0};
	double max_err = 0;
	double mean_err = 0;
	
	// loop through output data
	for(int i=0;i<N;++i){
		// put values in a temporary array for loop
		double d[] = {xOut[i][0]/N, xOut[i][1]/N, yOut[i][0]/N, yOut[i][1]/N};
		// loop through values
		for(int j=0;j<4;++j){
			// round the value and scale down by N to offset scaling from IFFT
			if(d[j] > 0){
				dataOut[i][j] = (char)(d[j]+0.5);
			}
			else{
				dataOut[i][j] = (char)(d[j]-0.5);
			}
			
			// save error statistics
			double err = d[j] - (double)dataIn[i][j];
//			errors[i][j] = err;
			if(err < 0) err = -err;
			if(err > max_err) max_err = err;
			mean_err += err/(N*4.0);
			
			// check value against initial value
			if(dataIn[i][j] != dataOut[i][j]){
				// will print if data does not match
				cout << "Values do not match: " << (int)dataIn[i][j] << ", " << (int)dataOut[i][j] << endl;
			}
		}
	}
	
	// print error statistics
	cout << endl << "Mean error: " << mean_err << endl;
	cout << "Max error: " << max_err << endl << endl;
//	cout << "first 25 errors: " << endl;
//	for(int i=0;i<25;++i){
//		for(int j=0;j<4;++j){
//			cout << errors[i][j] << ", ";
//		}
//		cout << endl;
//	}
}

