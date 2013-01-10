#include "svc.h"

namespace classifier{
	SVC::SVC(){
		std::cout << "Starting up Beelzebub's Support Vector Classifier...";
		this->SetupSvmParameter();

		std::cout << " Done." << std::endl ;
		std::cout << "LIBSVM version: " << libsvm_version << std::endl;
	}


	SVC::~SVC(){
		std::cout << "Exitting Support Vector Classifier." << std::endl;
		delete this->svm_problem_.x;
		delete this->svm_problem_.y;
	}


	void SVC::SetupSvmParameter(){
		std::cout << "Setting up parameters... ";

		this->svm_parameter_.svm_type		= NU_SVC;
		this->svm_parameter_.kernel_type	= RBF;
		this->svm_parameter_.degree			= 2;
		this->svm_parameter_.gamma			= 0.2;
		this->svm_parameter_.coef0			= 1;
		this->svm_parameter_.nu				= 0.2;
		this->svm_parameter_.cache_size		= 100;
		this->svm_parameter_.C				= 100;
		this->svm_parameter_.eps			= 0.00001;
//		this->svm_parameter_.eps			= 1e-3;
		this->svm_parameter_.p				= 0.1;
		this->svm_parameter_.shrinking		= 0.1;
		this->svm_parameter_.probability	= 0;
		this->svm_parameter_.nr_weight		= 0;
		this->svm_parameter_.weight_label	= NULL;

		std::cout << "Done." << std::endl;

		return;
	}


	void SVC::CheckSvmParameter(){
		std::string ret = "return value of svm_check_parameter.";

		std::cout << "Checking parameters... ";

		if(svm_check_parameter(&this->svm_problem_, &this->svm_parameter_) == NULL){
			std::cout << "Done. NO problem." << std::endl;
		}else{
			std::cerr << "Error! " << ret << std::endl;
		}

		return;
	}


	void SVC::InputTrainingData(){
		std::cout << "Inputting training data... ";

		this->training_data_[0][0].index = 1;
		this->training_data_[0][0].value = 0;
		this->training_data_[0][1].index = 2;
		this->training_data_[0][1].value = 2;
		this->training_data_[0][2].index = -1;

		this->training_data_[1][0].index = 1;
		this->training_data_[1][0].value = 0;
		this->training_data_[1][1].index = 2;
		this->training_data_[1][1].value = 4;
		this->training_data_[1][2].index = -1;
		
		this->training_data_[2][0].index = 1;
		this->training_data_[2][0].value = 1;
		this->training_data_[2][1].index = 2;
		this->training_data_[2][1].value = 3;
		this->training_data_[2][2].index = -1;

		this->training_data_[3][0].index = 1;
		this->training_data_[3][0].value = 2;
		this->training_data_[3][1].index = 2;
		this->training_data_[3][1].value = 4;
		this->training_data_[3][2].index = -1;


		this->training_data_[4][0].index = 1;
		this->training_data_[4][0].value = 2;
		this->training_data_[4][1].index = 2;
		this->training_data_[4][1].value = 0;
		this->training_data_[4][2].index = -1;

		this->training_data_[5][0].index = 1;
		this->training_data_[5][0].value = 3;
		this->training_data_[5][1].index = 2;
		this->training_data_[5][1].value = 1;
		this->training_data_[5][2].index = -1;
		
		this->training_data_[6][0].index = 1;
		this->training_data_[6][0].value = 4;
		this->training_data_[6][1].index = 2;
		this->training_data_[6][1].value = 0;
		this->training_data_[6][2].index = -1;

		this->training_data_[7][0].index = 1;
		this->training_data_[7][0].value = 4;
		this->training_data_[7][1].index = 2;
		this->training_data_[7][1].value = 1;
		this->training_data_[7][2].index = -1;

		std::cout << "Done." << std::endl;

		return;
	}


	void SVC::InputTestData(){
		std::cout << "Inputting test data... ";

/*
		this->test_data_[0].index = 1;
		this->test_data_[0].value = 1;
		this->test_data_[1].index = 2;
		this->test_data_[1].value = 4;
		this->test_data_[2].index = -1;
*/

		this->test_data_[0].index = 1;
		this->test_data_[0].value = 3;
		this->test_data_[1].index = 2;
		this->test_data_[1].value = 0;
		this->test_data_[2].index = -1;


		std::cout << "Done." << std::endl;

		return;
	}


	void SVC::BuildSvmProblem(){
		std::cout << "Building problems... ";

		this->svm_problem_.l = this->kSvmProblemNumber;
		this->svm_problem_.y = new double[this->svm_problem_.l];

		for(int i=0;i<this->svm_problem_.l;i++){
			// std::cout << this->svm_problem_.y[i];
			if(i < 4){
				this->svm_problem_.y[i] = 0;
			}else{
				this->svm_problem_.y[i] = 1;
			}
		}


		this->svm_problem_.x = new svm_node*[this->svm_problem_.l];
		for(int i=0;i<this->svm_problem_.l;i++){
			this->svm_problem_.x[i] = this->training_data_[i];
		}

		std::cout << "Done." << std::endl;

		return;
	}


	void SVC::BuildSvmModel(){
		int class_number = 0;

		std::cout << "Building model... ";

		this->svm_model_ = svm_train(&this->svm_problem_, &this->svm_parameter_);
		class_number =  svm_get_nr_class(this->svm_model_);

		std::cout << "Done." << std::endl;
		std::cout << "The number of class: " << class_number << std::endl;

		return;
	}


	void SVC::PredictResult(){
		this->result_ = svm_predict(this->svm_model_, this->test_data_);

		return;
	}


	double SVC::ClassifyTestData(){
		this->BuildSvmProblem();
		this->BuildSvmModel();
		this->PredictResult();

		#ifdef DEBUG
			std::cout << "***************************" << std::endl;
			std::cout << "Result of classification: " << this->result << std::endl;
			std::cout << "***************************" << std::endl;
		#endif

		return this->result_;
	}
}

