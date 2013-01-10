#ifndef __SVC_H__
#define __SVC_H__

#include "svm.h"

#include <iostream>
#include <string>
#include <stdlib.h>

namespace classifier{
	class SVC{
		private:
			static const int kSvmProblemNumber = 8;

			double result_;

			svm_parameter svm_parameter_;
			svm_problem svm_problem_;
			svm_model *svm_model_;
			struct svm_node training_data_[8][4];
			struct svm_node test_data_[3];

			void SetupSvmParameter();
			void CheckSvmParameter();
			void BuildSvmProblem();
			void BuildSvmModel();
			void PredictResult();
			void FinishSvm();

		public:
			SVC();
			~SVC();

			void InputTrainingData();
			void InputTestData();
			double ClassifyTestData();
	};
}

#endif /* __SVC_H__ */

