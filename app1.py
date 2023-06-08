#import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import math
import os

# Create multiple tabs
tabs = ["Bookings Visualization Tab", "Booking Cancellation Prediction Tab"]
active_tab = st.sidebar.radio("Select Tab", tabs)
absolute_path = os.path.dirname(os.path.abspath(__file__))

# Display content based on active tab
if active_tab == "Bookings Visualization Tab":
    #st.write("This is content for Visualization")
    #st.write("This is content for Visualization")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    ### set Heading for app
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Gain insights into the bookings, cancellations, and demographics of your hotel's customers</h2>
        </div>
        <br>
        """
    st.markdown(html_temp, unsafe_allow_html=True)

    ### Display a sample csv to the user.
    st.write("SAMPLE CSV:")
    st.write("Ensure that the CSV file you upload includes the same columns as the sample CSV file displayed below:")
    sample_dataset = pd.read_csv(absolute_path + '/sample.csv')
    st.dataframe(sample_dataset.head())

    ### Take csv file an input from user.
    # Create a file uploader in the Streamlit app
    st.write("Upload your CSV file below:")
    file = st.file_uploader("", type=["csv"])

    # If a file is uploaded
    if file is not None:
        # Read the CSV file using Pandas
        dataset = pd.read_csv(file)

        # Display the first few rows of your upload CSV file
        st.write("First few rows of your uploaded CSV file:")
        st.dataframe(dataset.head())

        # cleaning the dataset
        # 1
        # Checking for null/Missing/Nan values
        dataset1 = pd.isna(dataset)
        # Iterate over the columns in the dataframe
        for column in dataset1.columns:
            # Check if the column contains the value True
            if True in dataset1[column].values:
                # Print the name of the column
                print(column)

        # printing count of null values and null value percentages for each column
        null_count = dataset.isnull().sum()
        null_percent = null_count / len(dataset) * 100
        null_dataset = pd.DataFrame({'null_count': null_count, 'null_percent': null_percent})
        dataset = dataset.drop(['agent', 'company'], axis=1)
        # print(dataset.shape)
        # 2
        # drop the duplicate rows and retain only the one unique row
        dataset.drop_duplicates(inplace=True)
        print(dataset.shape)
        # 3
        # As we have object datatype for reservation_status_date column, we convert it to datetime format
        dataset['reservation_status_date'] = pd.to_datetime(dataset['reservation_status_date'])
        # 4
        # Encoding Categorical Data
        # 1: hotel -> City,Hotel - 0, Resort,Hotel - 1
        # Get the unique values in a column called "hotel"
        unique_values = dataset['hotel'].unique()
        # Create a LabelEncoder object and fit it to the unique values
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_values)
        # Apply the label encoding to the column
        dataset['hotel'] = label_encoder.transform(dataset['hotel'])
        # 2:Arrival_date_month -> 'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,'August': 8,'September': 9, 'October': 10, 'November': 11, 'December': 12
        # Get the unique values in a column called "arrival_date_month"
        unique_values = dataset['arrival_date_month'].unique()
        # Create a dictionary that maps month names to numerical values
        month_to_num = {
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9,
            'October': 10,
            'November': 11,
            'December': 12
        }
        # Replace month names with numerical values
        dataset['arrival_date_month'] = dataset['arrival_date_month'].map(month_to_num)
        # 3:market_segment: 0 -> Aviation, 1 -> Complementary, 2 ->  Corporate, 3 -> Direct, 4 -> Groups, 5 -> Offline TA / TO, 6 -> OnlineTA
        # Get the unique values in a column called "hotel"
        unique_values = dataset['market_segment'].unique()
        # Create a LabelEncoder object and fit it to the unique values
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_values)
        # Apply the label encoding to the column
        dataset['market_segment'] = label_encoder.transform(dataset['market_segment'])
        # 4:meal: 0-> BB, 1 -> FB, 2 -> HB, 3 -> SC, 4 -> Undefined
        # Get the unique values in a column called "hotel"
        unique_values = dataset['meal'].unique()
        # Create a LabelEncoder object and fit it to the unique values
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_values)
        # Apply the label encoding to the column
        dataset['meal'] = label_encoder.transform(dataset['meal'])

        # 5
        # Remove Unwanted columns
        # As marketing segment and distribution channel implying the same data we drop the distribution_channel column
        dataset = dataset.drop(['distribution_channel'], axis=1)

        # 6
        # Handling inconsistencies in the data.
        # So adult, babies , children cannot be zero because for every booking there must be atleast a single occupant, so I found the rows with all three attributes zero and removed those rows from dataset
        filter_a_b_c = (dataset.children == 0) & (dataset.adults == 0) & (dataset.babies == 0)
        # filter those zero valued rows from the dataset
        dataset = dataset[~filter_a_b_c]
        # 7
        # Data Integration
        # As babies and children nearly imply the same thing and we cannot get more insights by having two columns we will combine babies and children columns as Kids.
        # Combining children and babies column as Kids column
        dataset['Kids'] = dataset.children + dataset.babies
        # We don't have total guests count in the dataset so we make guests column by combining adult and kids column
        dataset['Guests'] = dataset.adults + dataset.Kids
        # 8
        # Handling outliers in the data.
        # we can see from the box plots and data description that many of the continuous features have outliers, so now let's treat those outliers by imputing values.
        dataset.loc[dataset.adults > 4, 'adults'] = 4
        dataset.loc[dataset.booking_changes > 5, 'booking_changes'] = 5
        dataset.loc[dataset.lead_time > 500, 'lead_time'] = 500
        dataset.loc[dataset.previous_bookings_not_canceled > 0, 'previous_bookings_not_canceled'] = 1
        dataset.loc[dataset.previous_cancellations > 0, 'previous_cancellations'] = 1
        dataset.loc[dataset.days_in_waiting_list > 0, 'days_in_waiting_list'] = 1
        dataset.loc[dataset.stays_in_weekend_nights >= 5, 'stays_in_weekend_nights'] = 5
        dataset.loc[dataset.stays_in_week_nights > 10, 'stays_in_week_nights'] = 10
        # 9
        # Rename the columns using a dictionary
        dataset = dataset.rename(columns={'arrival_date_year': 'arrival_year', 'arrival_date_month': 'arrival_month',
                                          'arrival_date_week_number': 'arrival_week_number',
                                          'stays_in_weekend_nights': 'weekend_night_stays',
                                          'stays_in_week_nights': 'week_night_stays',
                                          'days_in_waiting_list': 'waiting_days', 'Kids': 'kids', 'Guests': 'guests',
                                          'previous_cancellations': 'prev_cancel',
                                          'total_of_special_requests': 'special_requests',
                                          'previous_bookings_not_canceled': 'previous_uncancelled_bookings'})

        st.header("Visualizations of cancellations over different attributes")

        # Print the new column names
        # print(dataset.columns)
        # Visualizations:

        options = ['hotel', 'is_canceled', 'lead_time', 'arrival_month',
                   'arrival_week_number', 'arrival_date_day_of_month',
                   'weekend_night_stays', 'week_night_stays', 'meal', 'country', 'market_segment', 'is_repeated_guest',
                   'prev_cancel', 'previous_uncancelled_bookings', 'booking_changes', 'deposit_type', 'waiting_days',
                   'customer_type', 'adr', 'required_car_parking_spaces',
                   'special_requests', 'reservation_status', 'reservation_status_date',
                   'kids', 'guests']

        # 1
        # Extracting only cancelled bookings from the data set against the input paramter provided by user
        dataset_cancelled = dataset.loc[dataset['is_canceled'] == 1]

        st.header("Barplot analysis for cancelled bookings count against categorical parameters")


        def bar_plot():
            user_attribute = st.selectbox("Select an attribute for bar plot analysis", [None] + options)
            if user_attribute is not None:
                plt.figure(figsize=(14, 5))
                sns.countplot(x=user_attribute, data=dataset_cancelled, palette="crest", edgecolor="blue")
                plt.title("Cancellations trend over: " + user_attribute)
                st.pyplot()


        st.button("Bar plot", on_click=bar_plot())

        # 2
        # pie chart is used to observe the quantification of bookings count in different features
        st.header("Pie plot distribution of Booking count over categorical types - country & Market segments")
        features = ['customer_type', 'country']


        def pie_plot():
            feature = st.selectbox("Select an attribute for pie plot analysis", [None] + features)
            if feature is not None:
                custom_aggregation = {}
                custom_aggregation["arrival_date_day_of_month"] = "count"

                dataset_agg = dataset_cancelled.groupby(feature).agg(custom_aggregation)
                dataset_agg.columns = ["Booking Count"]
                dataset_agg[feature] = dataset_agg.index

                labels = dataset_agg[feature].tolist()
                values = dataset_agg['Booking Count'].tolist()

                pie_figure = px.pie(dataset_agg, values=values, names=labels)
                pie_figure.update_traces(textposition='inside')
                pie_figure.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
                pie_figure['layout'].update(height=550, width=500, title='Booking count by ' + feature, boxmode='group')
                # pie_figure.show()
                st.plotly_chart(pie_figure)


        st.button("Pie-plot", on_click=pie_plot())

        # 3
        st.header("Show Line plot analysis for average daily rate over time period")


        # # #Line plot, identify the relation between two features for different hotel types
        def line_plot():
            user_attribute = st.selectbox("Select an attribute for Line plot",
                                          [None] + ['lead_time', 'arrival_month', 'arrival_week_number',
                                                    'arrival_date_day_of_month'])
            if user_attribute is not None:
                # dataset_line = dataset_cancelled.loc[dataset_cancelled['is_repeated_guest'] == 1]
                plt.figure(figsize=(12, 7))
                sns.lineplot(x=dataset[user_attribute], y=dataset["adr"], marker="o", hue=dataset["hotel"])
                # sns.lineplot(x=dataset_line["prev_cancel"],y=dataset_cancelled["adr"],marker="o",hue=dataset_line["hotel"])
                # sns.lineplot(x=dataset["reserved_room_type"], y=dataset["adr"], marker="o", hue=dataset["hotel"])
                plt.title("Average Daily Rate trends");
                st.pyplot()


        st.button("Line plot", on_click=line_plot())

        # 4
        st.header("Violin plot of cancellations over deposit types")


        def violin_plot():

            user_attribute = st.selectbox("Select an attribute for violin plot", [None] + ['deposit_type', 'meal'])
            if user_attribute is not None:
                plt.figure(figsize=(12, 6))
                sns.violinplot(x=user_attribute, y='hotel', hue="is_canceled", data=dataset, bw=.2, cut=2, linewidth=2,
                               iner='box', split=True)
                sns.despine(left=True)
                plt.title(user_attribute + ' vs hotel vs cancellation', weight='bold')
                st.pyplot()


        st.button("Violin plot", on_click=violin_plot())

        # 5
        st.header("Show Cat plot analysis for cancellations over categorical input parameters")


        def cat_plot():
            user_attribute = st.selectbox("Select an attribute for Cat plot", [None] + options)
            if user_attribute is not None:
                st.write("Cat plot")
                sns.catplot(x=user_attribute, y="is_canceled", data=dataset, kind="point", height=4, aspect=2).set(
                    xlabel=user_attribute)
                plt.title("Cat plot")
                st.pyplot()


        st.button("Cat plot", on_click=cat_plot())


        # 6
        # We group the lead time and use leadtimes with >10 transactions for graph.
        st.header("Show Regression plot analysis for cancellations over input parameters")
        def plot():
            user_attribute = st.selectbox("Select an attribute",
                                          [None] + ["weekend_night_stays", "lead_time", "week_night_stays"])
            if user_attribute is not None:
                leadcancelleddata = dataset.groupby(user_attribute)["is_canceled"].describe()
                leadcancelleddatagrouped = leadcancelleddata.loc[leadcancelleddata["count"] >= 10]
                # Plotting the graph
                plt.figure(figsize=(14, 5))
                sns.regplot(x=leadcancelleddatagrouped.index, y=leadcancelleddatagrouped["mean"].values * 100,
                            color='blue')
                plt.title(user_attribute + " vs cancellations", fontsize=16)
                plt.xlabel(user_attribute, fontsize=16)
                plt.ylabel("cancellation_percent", fontsize=16)
                st.pyplot()


        st.button("Plot", on_click=plot())

else:
    #st.write("This is content for Tab 2")
    #open the pickle file created with randomforest_picklefile.py.
    pickle_in = open("rd_clf1.pkl","rb")
    rd_clf1=pickle.load(pickle_in)

    #@app.route('/')
    def welcome():
        return "Welcome All"

    #@app.route('/predict',methods=["Get"])

    #define and return the predict value using random forest classifier.
    def predict_note_authentication(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16):
        prediction=rd_clf1.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16]])
        print(p2)
        print(prediction)
        return prediction

    def main():
        #st.title("Hotel Bookings Cancellation prediction")
        #Create a header with orange background.
        html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h3 style="color:white;text-align:center;"> Booking Cancellation prediction using ML Model </h3>
        </div>
        """
        #Create input text fields
        st.markdown(html_temp,unsafe_allow_html=True)
        p1 = st.text_input("Please input your hotel type: For city hotels, enter '0', and for resort hotels, enter '1'.","")
        p2 = st.text_input("Please enter the number of days between the booking date and the arrival date.","")
        if p2:
            p2 = float(p2)
            p2 = math.log(p2)
        p3 = st.text_input("Please enter the number of weekday nights the guest will stay (Monday to Friday).","")
        p4 = st.text_input("Please enter the number of week nights the guest will stay (Saturday or Sunday).","")
        p5 = st.text_input("Plese enter the meal type value : Breakfast - 0, Full board -1, Half board - 2, , Room only-3, unknown - 4.","")
        p6 = st.text_input("Please enter the market segment type value where the booking is made from, for Aviation - 0, Complementary - 1, Corporate - 2, Direct - 3, Groups - 4 , Offline TA/TO - 5, Online TA- 6.","")
        p7 = st.text_input("Please enter the value '1' if guest stayed at the hotel before or if not enter '0'.","")
        p8 = st.text_input("Please the enter the number of previous cancellations the guest has made.","")
        p9 = st.text_input("please enter the number of  previous bookings the guest has made that were not canceled.","")
        p10 = st.text_input("Please enter the value of deposit type made for the booking : No Deposit - 0, Non Refundable - 1, Refundable - 2.","")
        p11 = st.text_input("Please enter the customer type value that made the booking: Contract - 0, Group - 1, Transient - 2, Group - 3","")
        p12 = st.text_input("Please enter the average daily rate given to the customer, which is the total revenue divided by the number of days booked.","")
        if p12:
            p12 = float(p12)
            p12 = math.log(p12)
        p13 = st.text_input("Please enter the number of parking spaces required by the guest.","")
        p14 = st.text_input("Please enter the total number of special requests made by the guest (e.g. extra towels, late check-out, etc.).","")
        p15 = st.text_input("Please enter the total number of kids (children and babies) in the books.","")
        p16 = st.text_input("Please enter the total number of Guests in the booking including kids.","")
        result=""
        if st.button("Predict"):
            result=predict_note_authentication(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16)
            if result == 0:
                result = "Booking will not be canceled"
            else:
                result = "Booking likely to be canceled"
        st.success('The prediction of the model is,  {}'.format(result))
        # if st.button("About"):
        #     st.text("Lets LEarn")
        #     st.text("Built with Streamlit")

    if __name__=='__main__':
        main()






