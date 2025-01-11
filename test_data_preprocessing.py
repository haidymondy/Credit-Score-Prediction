
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# test_data=pd.read_csv(r'C:\Users\Hazem\Desktop\awt\Data\test.csv')


class DataPreprocessor:
    def __init__(self):
     pass

    def drop_columns(self,data, columns):
        data.drop(columns, axis='columns', inplace=True)
        return data
   
    def clean_age(self,data):
        data["Age"] = data["Age"].str.replace("-", "", regex=False).str.replace("_", "", regex=False)
        data["Age"] = pd.to_numeric(data["Age"], downcast='integer', errors="coerce")
        data["Age"] = data["Age"].apply(lambda x: np.nan if x > 100 else x)
        data["Age"] = data.groupby("Customer_ID")["Age"].transform(lambda x: x.fillna(x.median()))
        data['Age'] = pd.cut(data['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle_Aged', 'Senior'])
        return data

    def clean_occupation(self,data):
        data["Occupation"] = data["Occupation"].replace("_______", "")
        data["Occupation"] = data.groupby("Customer_ID")["Occupation"].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "unemployed"))
        return data

    def clean_annual_income(self,data):
        data["Annual_Income"] = data["Annual_Income"].str.replace("_", "", regex=False)
        data["Annual_Income"] = pd.to_numeric(data["Annual_Income"], errors="coerce")
        data['Annual_Income_Category'] = pd.cut(data['Annual_Income'], bins=[0, 50000, 100000, np.inf], labels=['Low', 'Medium', 'High'])
        return data

    def clean_monthly_inhand_salary(self,data):
        data["Monthly_Inhand_Salary"] = data.groupby("Customer_ID")["Monthly_Inhand_Salary"].transform(
            lambda x: x.fillna(x.median()))
        data["Monthly_Inhand_Salary"].fillna(data["Monthly_Inhand_Salary"].median(), inplace=True)
        return data

    def clean_num_of_loan(self,data):
        data["Num_of_Loan"] = data["Num_of_Loan"].str.replace("-", "", regex=False).str.replace("_", "", regex=False).str.replace("-100", "0", regex=False)
        data["Num_of_Loan"] = pd.to_numeric(data["Num_of_Loan"], downcast='integer', errors="coerce")
        return data

    def clean_type_of_loan(self,data):
        def process_cell(value):
            if pd.isna(value):
                return "Not Specified"
            elif "," in value:
                return "Other"
            else:
                return value.strip()

        data["Type_of_Loan"] = data["Type_of_Loan"].apply(process_cell)
        return data

    def clean_num_of_delayed_payment(self,data):
        data["Num_of_Delayed_Payment"] = data["Num_of_Delayed_Payment"].str.replace("_", "", regex=False)
        data["Num_of_Delayed_Payment"] = pd.to_numeric(data["Num_of_Delayed_Payment"], errors="coerce")
        data["Num_of_Delayed_Payment"] = data.groupby("Customer_ID")["Num_of_Delayed_Payment"].transform(lambda x: x.fillna(x.median()))
        data["Num_of_Delayed_Payment"].fillna(data["Num_of_Delayed_Payment"].median(), inplace=True)

        return data

    def clean_changed_credit_limit(self,data):
        data["Changed_Credit_Limit"] = data["Changed_Credit_Limit"].str.replace("_", "", regex=False)
        data["Changed_Credit_Limit"] = pd.to_numeric(data["Changed_Credit_Limit"], errors="coerce")
        data["Changed_Credit_Limit"] = data.groupby("Customer_ID")["Changed_Credit_Limit"].transform(lambda x: x.fillna(x.median()))
        return data

    def clean_credit_mix(self,data):
        data["Credit_Mix"] = data["Credit_Mix"].replace("_", np.nan)
        data["Credit_Mix"] = data.groupby("Customer_ID")["Credit_Mix"].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Standard"))
        return data

    def clean_outstanding_debt(self,data):
        data["Outstanding_Debt"] = data["Outstanding_Debt"].replace("_", "", regex=True)
        data["Outstanding_Debt"] = pd.to_numeric(data["Outstanding_Debt"], errors="coerce")
        return data

    def clean_credit_history_age(self,data):
        data["Credit_History_Age"] = data.groupby("Customer_ID")["Credit_History_Age"].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "0 Years and 0 Months"))

        def convert_to_months(age_str):
            match = re.match(r"(\d+)\sYears\sand\s(\d+)\sMonths", age_str)
            if match:
                years = int(match.group(1))
                months = int(match.group(2))
                return years * 12 + months
            return np.nan

        data["Credit_History_Age"] = data["Credit_History_Age"].apply(convert_to_months)
        return data

    def clean_payment_of_min_amount(self,data):
        data["Payment_of_Min_Amount"] = data["Payment_of_Min_Amount"].replace("NM", "No")
        return data

    def clean_amount_invested_monthly(self,data):
        data["Amount_invested_monthly"] = data["Amount_invested_monthly"].replace("__10000__", "1000")
        data["Amount_invested_monthly"] = pd.to_numeric(data["Amount_invested_monthly"], errors="coerce").round(3)
        data["Amount_invested_monthly"] = data.groupby("Customer_ID")["Amount_invested_monthly"].transform(lambda x: x.fillna(x.mean()))
        return data

    def clean_payment_behaviour(self,data):
        data["Payment_Behaviour"] = data["Payment_Behaviour"].replace("!@9#%8", np.nan)
        overall_mode = data["Payment_Behaviour"].mode()[0]
        data["Payment_Behaviour"] = data.groupby("Customer_ID")["Payment_Behaviour"].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else overall_mode))
        return data

    def clean_monthly_balance(self,data):
        data["Monthly_Balance"] = data["Monthly_Balance"].replace("__", "", regex=True)
        data["Monthly_Balance"] = pd.to_numeric(data["Monthly_Balance"], errors="coerce").round(3)
        data["Monthly_Balance"] = data.groupby("Customer_ID")["Monthly_Balance"].transform(lambda x: x.fillna(x.median()))
        data["Monthly_Balance"].fillna(data["Monthly_Balance"].median(), inplace=True)

        return data
    
    def remove_outliers_iqr(self, data, column):
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    def clip_values(self, data, column, lower_bound=None, upper_bound=None):
        data[column] = np.clip(data[column], lower_bound, upper_bound)



    def preprocess_data(self,data):
        data = self.clean_age(data)
        data = self.clean_occupation(data)
        data = self.clean_annual_income(data)
        data = self.clean_monthly_inhand_salary(data)
        data = self.clean_num_of_loan(data)
        data = self.clean_type_of_loan(data)
        data = self.clean_num_of_delayed_payment(data)
        data = self.clean_changed_credit_limit(data)
        data = self.clean_credit_mix(data)
        data = self.clean_outstanding_debt(data)
        data = self.clean_credit_history_age(data)
        data = self.clean_payment_of_min_amount(data)
        data = self.clean_amount_invested_monthly(data)
        data = self.clean_payment_behaviour(data)
        data = self.clean_monthly_balance(data)

        # Remove outliers
        data = self.remove_outliers_iqr(data, 'Annual_Income')
        data = self.remove_outliers_iqr(data, 'Interest_Rate')
        data = self.remove_outliers_iqr(data, 'Total_EMI_per_month')

        # Clip values
        self.clip_values(data, 'Num_Bank_Accounts', None, 15)
        self.clip_values(data, 'Num_Credit_Card', None, 20)
        self.clip_values(data, 'Num_of_Loan', None, 150)
        self.clip_values(data, 'Num_of_Delayed_Payment', None, 200)
        self.clip_values(data, 'Changed_Credit_Limit', None, 35)
        self.clip_values(data, 'Monthly_Balance', 0, None)

        columns_to_drop = ['Name', 'SSN', 'ID', 'Num_Credit_Inquiries','Customer_ID']
        self.drop_columns(data, columns_to_drop)
      
        return data





class DataEncoder:
    def __init__(self):
        self.cols_to_scale = [
            'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Card',
            'Interest_Rate', 'Outstanding_Debt', 'Credit_History_Age',
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
            'Delay_from_due_date', 'Num_of_Delayed_Payment','Credit_Utilization_Ratio'
        ]
        
        self.scaler = MinMaxScaler()

        self.encoders = {
            'Credit_Mix': [['Bad', 'Standard', 'Good']],
            'Age': [['Young', 'Middle_Aged', 'Senior']],
            'Annual_Income_Category': [['Low', 'Medium', 'High']],
            'Payment_Behaviour': [
                ['Low_spent_Small_value_payments', 'Low_spent_Medium_value_payments',
                 'Low_spent_Large_value_payments', 'High_spent_Small_value_payments',
                 'High_spent_Medium_value_payments', 'High_spent_Large_value_payments']
            ],
            'Month' :[[ 'January', 'February', 'March', 'April', 'May',
            'June', 'July', 'August','September', 'October','November', 'December']]
        }

        self.columns_to_encode = ['Type_of_Loan', 'Occupation', 'Payment_of_Min_Amount']

        self.label_encoder = LabelEncoder()

    def apply_min_max_scaling(self, data):
        data[self.cols_to_scale] = self.scaler.fit_transform(data[self.cols_to_scale])

    def apply_ordinal_encoding(self, data):
        for column, categories in self.encoders.items():
            encoder = OrdinalEncoder(categories=categories)
            data[column] = encoder.fit_transform(data[[column]])

    def apply_label_encoding(self, data):
        for column in self.columns_to_encode:
            data[column] = self.label_encoder.fit_transform(data[column])


    def preprocess(self, data):
        self.apply_min_max_scaling(data)
        self.apply_ordinal_encoding(data)
        self.apply_label_encoding(data)
        return data


# preprocessor1 = DataPreprocessor()
# test_data = preprocessor1.preprocess_data(test_data)


# preprocessor2 = DataEncoder()
# data = preprocessor2.preprocess(test_data)


