import sys

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, \
    QPushButton, QLabel, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem, QMessageBox
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.style.use('seaborn-poster')


# Read the CSV file into a DataFrame
df = pandas.read_csv('desurvey.csv')

# Access the data in the DataFrame
data = df.to_numpy()
print(df.shape)

list = ['BHID', 'FROM', 'TO', '_len', 'CU', '_acum', 'azmm', 'dipm', 'xm', 'ym', 'zm', 'azmb', 'dipb', 'xb', 'yb', 'zb', 'azme', 'dipe', 'xe', 'ye', 'ze', 'BHIDint']


class Widget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Machine Learning")
        self.setSizeIncrement(300, 300)

        self.vLayout = QVBoxLayout()
        self.hLayout = QHBoxLayout()

        self.visualize_btn = QPushButton("Visualize data")
        self.logistic_btn = QPushButton("Logistic Regression")
        self.k_nearest_btn = QPushButton("K-Nearest Neighbors")
        self.svm_btn = QPushButton("SVM")
        self.kernel_svm_btn = QPushButton("Kernel SVM")
        self.naive_bayes_btn = QPushButton("Naive Bayes")
        self.decision_tree_btn = QPushButton("Decision Tree")
        self.random_forest_classify_btn = QPushButton("Random Forest Classification")
        self.hLayout.addWidget(self.visualize_btn)
        self.hLayout.addWidget(self.logistic_btn)
        self.hLayout.addWidget(self.k_nearest_btn)
        self.hLayout.addWidget(self.svm_btn)
        self.hLayout.addWidget(self.kernel_svm_btn)
        self.hLayout.addWidget(self.naive_bayes_btn)
        self.hLayout.addWidget(self.decision_tree_btn)
        self.hLayout.addWidget(self.random_forest_classify_btn)

        self.vLayout.addLayout(self.hLayout)

        self.hLayout_section = QHBoxLayout()
        self.vLayout_section = QVBoxLayout()

        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(df.shape[0])
        self.table_widget.setColumnCount(df.shape[1])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data[i][j]))
                self.table_widget.setItem(i, j, item)

        # ComboBox
        self.x_axis1_label = QLabel("X_1")
        self.x_value1_combo = QComboBox()
        for item in list:
            self.x_value1_combo.addItem(item)

        self.x_axis2_label = QLabel("X_2")
        self.x_value2_combo = QComboBox()
        for item in list:
            self.x_value2_combo.addItem(item)

        self.x_axis3_label = QLabel("X_3")
        self.x_value3_combo = QComboBox()
        for item in list:
            self.x_value3_combo.addItem(item)

        self.y_axis_label = QLabel("Y")
        self.y_value_combo = QComboBox()
        for item in list:
            self.y_value_combo.addItem(item)

        self.test_size_label = QLabel("test_size")
        self.test_size_edit = QLineEdit("0.25")
        self.random_state_label = QLabel("random_state")
        self.random_state_edit = QLineEdit("0")
        self.p_label = QLabel("P")
        self.p_edit = QLineEdit("2")
        self.n_neighbors_label = QLabel("n_neighbors")
        self.n_neighbors_edit = QLineEdit("5")
        self.n_estimator_label = QLabel("n_estimator")
        self.n_estimator_edit = QLineEdit("10")

        self.value = QPushButton("Value")
        self.apply = QPushButton("Apply")

        self.x_value1_combo.setCurrentIndex(8)
        self.x_value2_combo.setCurrentIndex(9)
        self.x_value3_combo.setCurrentIndex(10)
        self.y_value_combo.setCurrentIndex(21)
        self.hLayout_section.addWidget(self.test_size_label)
        self.hLayout_section.addWidget(self.test_size_edit)
        self.hLayout_section.addWidget(self.random_state_label)
        self.hLayout_section.addWidget(self.random_state_edit)
        self.hLayout_section.addWidget(self.p_label)
        self.hLayout_section.addWidget(self.p_edit)
        self.hLayout_section.addWidget(self.n_neighbors_label)
        self.hLayout_section.addWidget(self.n_neighbors_edit)
        self.hLayout_section.addWidget(self.n_estimator_label)
        self.hLayout_section.addWidget(self.n_estimator_edit)
        self.hLayout_section.addWidget(self.x_axis1_label)
        self.hLayout_section.addWidget(self.x_value1_combo)
        self.hLayout_section.addWidget(self.x_axis2_label)
        self.hLayout_section.addWidget(self.x_value2_combo)
        self.hLayout_section.addWidget(self.x_axis3_label)
        self.hLayout_section.addWidget(self.x_value3_combo)
        self.hLayout_section.addWidget(self.y_axis_label)
        self.hLayout_section.addWidget(self.y_value_combo)

        self.hLayout_section.addWidget(self.apply)
        self.hLayout_section.addWidget(self.value)

        self.vLayout_section.addWidget(self.table_widget)
        self.vLayout_section.addLayout(self.hLayout_section)
        self.vLayout.addLayout(self.vLayout_section)
        self.setLayout(self.vLayout)

        # Signal and slot
        self.visualize_btn.clicked.connect(self.visualize_data)
        self.k_nearest_btn.clicked.connect(self.k_nearest)
        self.logistic_btn.clicked.connect(self.logistic_regression)
        self.svm_btn.clicked.connect(self.svm)
        self.kernel_svm_btn.clicked.connect(self.kernel_svm)
        self.naive_bayes_btn.clicked.connect(self.naive_bayes)
        self.decision_tree_btn.clicked.connect(self.decision_tree)
        self.random_forest_classify_btn.clicked.connect(self.random_forest_classify)

        # QMessageBox.information(self, str("Success"), str("You are loaded csv file successfully."), QMessageBox.Ok)

        self.show()

    def visualize_data(self):
        x1_value = self.x_value1_combo.currentText()
        x2_value = self.x_value2_combo.currentText()
        x3_value = self.x_value3_combo.currentText()
        y_value = self.y_value_combo.currentText()
        x_1 = data[:, list.index(x1_value)]
        x_2 = data[:, list.index(x2_value)]
        x_3 = data[:, list.index(x3_value)]
        y = data[:, list.index(y_value)]

        X = np.array([x_1, x_2, x_3]).T
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')
        X_train = X_train.astype('float64')
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.grid()

        ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c='r', s=50)

        ax.set_title('Training data visualizing')

        # # Set axes label
        ax.set_xlabel(x1_value, labelpad=20)
        ax.set_ylabel(x2_value, labelpad=20)
        ax.set_zlabel(x3_value, labelpad=20)
        plt.show()

    def logistic_regression(self):
        x1_value = self.x_value1_combo.currentText()
        x2_value = self.x_value2_combo.currentText()
        x3_value = self.x_value3_combo.currentText()
        y_value = self.y_value_combo.currentText()
        x_1 = data[:, list.index(x1_value)]
        x_2 = data[:, list.index(x2_value)]
        x_3 = data[:, list.index(x3_value)]
        y = data[:, list.index(y_value)][300:700]

        X = np.array([x_1[300:700], x_2[300:700]]).T

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        X_test = X_test.astype('float64')
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)

        # Training the Logistic Regression model on the Training set
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)

        # Predicting a new result
        print(classifier.predict(sc.transform([[30, 87000]])))

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

        # Visualising the Training set results
        X_set, y_set = sc.inverse_transform(X_train), y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25), np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(121)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-5, 1700, 'Model Score : ' + str(accuracy_score(y_test, y_pred)), fontsize=12, color='red')
        plt.legend()

        # Visualising the Test set results
        X_set, y_set = sc.inverse_transform(X_test), y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25), np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(122)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.suptitle('Logistic Regression (Training and Test set)')
        plt.text(2.2940, 422500, 'Confusion_matrix : ' + str(cm), fontsize=12, color='red')
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.legend()
        plt.show()

    def svm(self):
        x1_value = self.x_value1_combo.currentText()
        x2_value = self.x_value2_combo.currentText()
        x3_value = self.x_value3_combo.currentText()
        y_value = self.y_value_combo.currentText()
        x_1 = data[:, list.index(x1_value)]
        x_2 = data[:, list.index(x2_value)]
        x_3 = data[:, list.index(x3_value)]
        y = data[:, list.index(y_value)][300:500]

        X = np.array([x_1[300:500], x_2[300:500]]).T

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        X_test = X_test.astype('float64')
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)

        # Training the SVM model on the Training set
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(X_train, y_train)

        # Predicting a new result
        print(classifier.predict(sc.transform([[30, 87000]])))

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

        # Visualising the Training set results
        X_set, y_set = sc.inverse_transform(X_train), y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25), np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(121)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-5, 1700, 'Model Score : ' + str(accuracy_score(y_test, y_pred)), fontsize=12, color='red')
        plt.legend()

        # Visualising the Test set results
        X_set, y_set = sc.inverse_transform(X_test), y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25), np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(122)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.suptitle('SVM (Training and Test set)')
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-10, 1700, 'Confusion_matrix : ' + str(cm), fontsize=12, color='red')
        plt.legend()
        plt.show()

    def kernel_svm(self):
        x1_value = self.x_value1_combo.currentText()
        x2_value = self.x_value2_combo.currentText()
        x3_value = self.x_value3_combo.currentText()
        y_value = self.y_value_combo.currentText()
        x_1 = data[:, list.index(x1_value)]
        x_2 = data[:, list.index(x2_value)]
        x_3 = data[:, list.index(x3_value)]
        y = data[:, list.index(y_value)][300:500]

        X = np.array([x_1[300:500], x_2[300:500]]).T

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        X_test = X_test.astype('float64')
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)

        # Training the SVM model on the Training set
        classifier = SVC(kernel='rbf', random_state=0)
        classifier.fit(X_train, y_train)

        # Predicting a new result
        print(classifier.predict(sc.transform([[30, 87000]])))

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

        # Visualising the Training set results
        X_set, y_set = sc.inverse_transform(X_train), y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                             np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(121)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-5, 1700, 'Model Score : ' + str(accuracy_score(y_test, y_pred)), fontsize=12, color='red')
        plt.legend()

        # Visualising the Test set results
        X_set, y_set = sc.inverse_transform(X_test), y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                             np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(122)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.suptitle('Kernel SVM (Training and Test set)')
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-10, 1700, 'Confusion_matrix : ' + str(cm), fontsize=12, color='red')
        plt.legend()
        plt.show()

    def naive_bayes(self):
        x1_value = self.x_value1_combo.currentText()
        x2_value = self.x_value2_combo.currentText()
        x3_value = self.x_value3_combo.currentText()
        y_value = self.y_value_combo.currentText()
        x_1 = data[:, list.index(x1_value)]
        x_2 = data[:, list.index(x2_value)]
        x_3 = data[:, list.index(x3_value)]
        y = data[:, list.index(y_value)][300:500]

        X = np.array([x_1[300:500], x_2[300:500]]).T

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        X_test = X_test.astype('float64')
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)

        # Training the Naive Bayes model on the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # Predicting a new result
        print(classifier.predict(sc.transform([[30, 87000]])))

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = sc.inverse_transform(X_train), y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                             np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(121)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-5, 1700, 'Model Score : ' + str(accuracy_score(y_test, y_pred)), fontsize=12, color='red')
        plt.legend()

        # Visualising the Test set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = sc.inverse_transform(X_test), y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                             np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(122)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.suptitle('Naive Bayes (Training and Test set)')
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-10, 1700, 'Confusion_matrix : ' + str(cm), fontsize=12, color='red')
        plt.legend()
        plt.show()

    def decision_tree(self):
        x1_value = self.x_value1_combo.currentText()
        x2_value = self.x_value2_combo.currentText()
        x3_value = self.x_value3_combo.currentText()
        y_value = self.y_value_combo.currentText()
        x_1 = data[:, list.index(x1_value)]
        x_2 = data[:, list.index(x2_value)]
        x_3 = data[:, list.index(x3_value)]
        y = data[:, list.index(y_value)][300:500]

        X = np.array([x_1[300:500], x_2[300:500]]).T

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        X_test = X_test.astype('float64')
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)

        # Training the Decision Tree Classification model on the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)

        # Predicting a new result
        print(classifier.predict(sc.transform([[30, 87000]])))

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

        # Visualising the Training set results
        X_set, y_set = sc.inverse_transform(X_train), y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25), np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(121)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-5, 1700, 'Model Score : ' + str(accuracy_score(y_test, y_pred)), fontsize=12, color='red')
        plt.legend()

        # Visualising the Test set results
        X_set, y_set = sc.inverse_transform(X_test), y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                             np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.subplot(122)
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('grey', 'white')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue'))(i), label=j)
        plt.suptitle('Decision Tree Classification (Training and Test set)')
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-10, 1700, 'Confusion_matrix : ' + str(cm), fontsize=12, color='red')
        plt.legend()
        plt.show()

    def k_nearest(self):
        x1_value = self.x_value1_combo.currentText()
        x2_value = self.x_value2_combo.currentText()
        x3_value = self.x_value3_combo.currentText()
        y_value = self.y_value_combo.currentText()
        x_1 = data[:, list.index(x1_value)]
        x_2 = data[:, list.index(x2_value)]
        x_3 = data[:, list.index(x3_value)]
        y = data[:, list.index(y_value)][300:500]

        X = np.array([x_1[300:500], x_2[300:500]]).T

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        X_test = X_test.astype('float64')
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)

        # Training the K-NN model on the Training set
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(X_train, y_train)

        # Predicting a new result
        print(classifier.predict(sc.transform([[30, 87000]])))

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

        # Visualising the Training set results
        X_set, y_set = sc.inverse_transform(X_train), y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1), np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1))
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('grey', 'white', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue', 'red'))(i), label=j)
        plt.title('K-NN (Training set)')
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-5, 1700, 'Model Score : ' + str(accuracy_score(y_test, y_pred)), fontsize=12, color='red')
        plt.legend()
        plt.show()

        # Visualising the Test set results
        X_set, y_set = sc.inverse_transform(X_test), y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1), np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1))
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('grey', 'white', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('black', 'blue', 'red'))(i), label=j)
        plt.title('K-NN (Test set)')
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.text(-10, 1700, 'Confusion_matrix : ' + str(cm), fontsize=12, color='red')
        plt.legend()
        plt.show()

    def random_forest_classify(self):
        x1_value = self.x_value1_combo.currentText()
        x2_value = self.x_value2_combo.currentText()
        x3_value = self.x_value3_combo.currentText()
        y_value = self.y_value_combo.currentText()
        x_1 = data[:, list.index(x1_value)]
        x_2 = data[:, list.index(x2_value)]
        x_3 = data[:, list.index(x3_value)]
        y = data[:, list.index(y_value)][300:500]

        X = np.array([x_1[300:500], x_2[300:500]]).T

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        X_test = X_test.astype('float64')
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)

        # Training the Random Forest Classification model on the Training set
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)

        # Predicting a new result
        print(classifier.predict(sc.transform([[30, 87000]])))

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = sc.inverse_transform(X_train), y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                             np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Random Forest Classification (Training set)')
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.legend()
        plt.show()

        # Visualising the Test set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = sc.inverse_transform(X_test), y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                             np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Random Forest Classification (Test set)')
        plt.xlabel(x1_value)
        plt.ylabel(x2_value)
        plt.legend()
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Widget()
    window.show()

    app.exec()
