# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t

# Modules
from src.linear_regression_dashboard.get_data import FredData

# Plots
from src.linear_regression_dashboard.plots import TwoLinesPlot
from src.linear_regression_dashboard.plots import AdjustableScatterPlot
from src.linear_regression_dashboard.plots import SSEPlot


def OLSExampleLayout():

    # Define the Dependent Variable
    y = FredData('UNRATE')
    y.index = pd.to_datetime(y.index)
    y.index = y.index.strftime('%Y-%m')

    # Define the Independent Variable
    x = FredData('GDPC1')
    x.index = pd.to_datetime(x.index)
    x.index = x.index.strftime('%Y-%m')

    # Define the DataFrame
    ols_df = pd.DataFrame(index=y.index)
    ols_df['unemployment_rate'] = y
    ols_df['economic_growth_rate'] = x.pct_change(4).mul(100).round(2)
    ols_df = ols_df.dropna()

    ols_df['constant'] = 1

    st.divider()

    st.subheader('Check the Data')

    col_right, buff, col_left = st.columns([3.5, 0.5, 6])

    with col_right:
        st.dataframe(ols_df, height=550)

    with col_left:
        fig1 = TwoLinesPlot(
            ols_df['unemployment_rate'],
            ols_df['economic_growth_rate'],
            'Unemployment Rate and Economic Growth Time Series',
            'Unemployment Rate',
            'Economic Growth Rate'
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    st.divider()

    st.subheader('Check the OLS Logic')

    st.text(
        """
        When scattering the data and try to fit an estimated trend line you can observe this will never be the perfect
        fill for the real data. Then, there will always be this 'errors' on our estimation, the logic behind the 
        Ordinary Least Squares (OLS) is to minimize this errors by optimizing the Squared Sum of Residuals (or Errors).
        
        There is an infinite combination of betas, but just one minimizes the SSE. Here you can see a very graphic
        example:
        """
    )

    b0_slider = st.slider("Select a value for b0", -10.0, 10.0, 0.0)

    b1_slider = st.slider("Select a value for b1", -1.0, 1.0, 0.0)

    col1, col2 = st.columns([5, 5])

    with col1:
        fig2 = AdjustableScatterPlot(
            ols_df['unemployment_rate'],
            ols_df['economic_growth_rate'],
            b0_slider,
            b1_slider,
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    with col2:
        fig3 = SSEPlot(
                ols_df['unemployment_rate'],
                ols_df['economic_growth_rate'],
                b0_slider,
                b1_slider,
            )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    st.divider()

    st.title('The Matrix Algebra')

    col1, col2 = st.columns([5, 5])

    with col1:
        st.header("Algebraic Form")

        st.latex(r'''
        b_1 = \frac{n \sum (xy) - \sum x \sum y}{n \sum x^2 - (\sum x)^2}
        ''')

        st.latex(r'''
        b_0 = \bar{y} - b_1 \bar{x}
        ''')

    with col2:
        st.header("Matrix Form")
        st.latex(r'''
        \mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}
        ''')

        st.latex(r'''
        \boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
        ''')

    st.header("Vectors and Matrix")

    st.text(
        """
        Information Matrix is very important to calculate the OLS regression. Information Matrix must be invertible in 
        order to calculate the MCO algorithm. The Matrix is only invertible if its rank is equal to the number of 
        columns. But what is the 'Rank?'. What if the Rank < number of columns?. What is a singular matrix?.
        """
    )

    Information_Matrix = ols_df[['constant', 'economic_growth_rate']].to_numpy()

    st.text(
        f"""
        This Matrix Rank is: {np.linalg.matrix_rank(Information_Matrix)}
        """
    )

    st.text(
        """
        Since the Rank is equal to the Information Matrix's number of columns, out Matrix is invertible and the betas
        can be calculated.
        """
    )

    col3, col4 = st.columns([5, 5])

    X_matrix = r'\mathbf{X} = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}'
    X_values = (r'\mathbf{X} = \begin{bmatrix} 1 & ' +
                f'{ols_df['economic_growth_rate'][0]:.2f}' + r' \\ 1 & ' +
                f'{ols_df['economic_growth_rate'][1]:.2f}' + r' \\ \vdots & \vdots \\ 1 & ' +
                f'{ols_df['economic_growth_rate'][-1]:.2f}' + r' \end{bmatrix}')

    y_vector = r'\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}'
    y_values = (r'\mathbf{y} = \begin{bmatrix} ' +
                f'{ols_df['unemployment_rate'][0]:.2f}' + r' \\ '
                + f'{ols_df['unemployment_rate'][1]:.2f}' + r' \\ \vdots \\ ' +
                f'{ols_df['unemployment_rate'][-1]:.2f}' + r' \end{bmatrix}')

    with col3:
        st.subheader("Dependent Variable Vector")
        st.latex(y_vector)
        st.latex(y_values)

    with col4:
        st.subheader("Information Matrix")
        st.latex(X_matrix)
        st.latex(X_values)

    st.header("The Calculations")

    st.text(
        """
        When you multiply a Matrix by its transposed form it is equivalent to square the Matrix; this new Matrix has 
        interesting properties that we might check later
    
        To obtain (X'X) we should, first transpose the Information Matrix and then multiply it by the original. This
        Matrix can also be interpreted as a kind of 'Variance' for X.
        """
    )

    col1, col2 = st.columns([3, 7])

    with col1:
        st.write('Formula')
        st.latex(r'''
        \mathbf{X}^\top \mathbf{X} = \begin{bmatrix}
        \sum 1 & \sum x \\
        \sum x & \sum x^2
        \end{bmatrix}
        ''')

    with col2:
        st.write('Data')
        Information_Matrix_T = Information_Matrix.transpose()
        Information_Matrix_Square = Information_Matrix_T.dot(Information_Matrix)

        st.table(Information_Matrix_Square)

    with st.expander("This is an Algebraic Tip..."):
        st.text(
            """
            Multiplication in a Matrix Algebra problem is not as in Classic Algebra. To Multiply two Matrices A * B, A 
            has to be the same number of columns than B's rows'.
        
            if A is nxm B must be mxk so they can be multiplied; the result Matrix is going to be a nxk Matrix
            """
        )

    st.text(
        """
        This matrix must be invertible for the OLS betas calculation, since the Rank is equal to the number of columns,
        the Square of the Information Matrix is invertible.
        
        Now let's obtain the Inverse of the Squared Matrix Information (X'X)^(-1). This Matrix will mantain the same 
        dimensions as the original Matrix:
        """
    )

    col1, col2 = st.columns([3, 7])

    with col1:
        st.write('Formula')
        st.latex(r'''(\mathbf{X}^\top \mathbf{X})^{-1} = \frac{\text{adj}(\mathbf{X}^\top \mathbf{X})}{\det(\mathbf{
        X}^\top \mathbf{X})} ''')

    with col2:
        st.write('Data')
        X_Variance_Matrix_Inverse = np.linalg.inv(Information_Matrix_Square)

        st.table(X_Variance_Matrix_Inverse)

    st.text(
        """
        Now we obtain the Multiplication of X'*Y (we use X Transposed) which can be interpreted as the Covariance of
        Y and the X's.
        """
    )

    col1, col2 = st.columns([3, 7])

    with col1:
        st.write('Formula')
        st.latex(r'''
        \mathbf{X}^\top \mathbf{y} = \begin{bmatrix}
        \sum y \\
        \sum xy
        \end{bmatrix}
        ''')

    with col2:
        st.write('Data')
        Y_Vector = ols_df["unemployment_rate"].to_numpy()

        Y_Covariance_X = Information_Matrix_T.dot(Y_Vector)

        st.table(Y_Covariance_X)

    st.text(
        """
        Mathematically the Betas are equivalent to the a quotient that explains how much the variation of Y is explained 
        by the variation of X. Other form to understand its formula is:
        """
    )

    col1, col2 = st.columns([3, 7])

    with col1:
        st.write('Formula')
        st.latex(r'''
        \beta_1 = \frac{\text{Cov}(x,y)}{\text{Var}(x)}
        ''')

    with col2:
        st.write('Data')
        Beta = X_Variance_Matrix_Inverse.dot(Y_Covariance_X)

        st.table(Beta)

    st.text(
        f"""
        This is why it makes kind of sense to understand the Squared Information Matrix as the Variance of X, we invert
        it because it must go in the denominator. And the Vector of X * Y is understand as the Covariance between Y and
        the X's.
        
        The we see that the B0 is {Beta[0].round(4)} and the B1 is {Beta[1].round(4)} just as we observed above.
        """
    )

    st.divider()

    st.title('The Fitted Values')

    st.text(
        """
        Let's now estimate the fitted values of our regression model. This are the estimations of our dependent variable
        using the betas we obtained from the OLS algorithm.
        """
    )

    ols_df['fitted_values'] = Information_Matrix.dot(Beta)

    col_right, buff, col_left = st.columns([3.5, 0.5, 6])

    with col_right:
        st.dataframe(ols_df[['unemployment_rate', 'fitted_values']], height=550)

    with col_left:
        fig4 = TwoLinesPlot(
            ols_df['unemployment_rate'],
            ols_df['fitted_values'],
            'Real Values vs Fitted Values',
            'Unemployment Rate',
            'Estimated Unemployment Rate',
            False
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    st.text(
        """
        Fitted Values highly depend on the betas and this determine how 'close' our model is to reality.
        """
    )

    st.text(
        """
        A bad calculation of the betas can also produce a miscalculation on the error terms.
        """
    )

    b0_slider_2 = st.slider("Select a value for b0", -10.0, 10.0, 6.4294, key='b0_slider')

    b1_slider_2 = st.slider("Select a value for b1", -1.0, 1.0, -0.2279, key='b1_slider')

    ols_df['inferred_fitted_values'] = b0_slider_2 + b1_slider_2 * ols_df['economic_growth_rate']

    col1, col2 = st.columns([5, 5])

    with col1:
        fig5 = TwoLinesPlot(
            ols_df['fitted_values'],
            ols_df['inferred_fitted_values'],
            'OLS Fitted Values vs Inferred Fitted Values',
            'OLS Fitted Values',
            'Inferred Fitted Values',
            False
        )
        st.plotly_chart(
            fig5,
            use_container_width=True
        )

    with col2:
        ols_df['error_terms'] = ols_df['unemployment_rate'] - ols_df['fitted_values']

        ols_df['inferred_error_terms'] = ols_df['unemployment_rate'] - ols_df['inferred_fitted_values']

        fig6 = TwoLinesPlot(
            ols_df['error_terms'],
            ols_df['inferred_error_terms'],
            'OLS Residuals vs Inferred Residuals',
            'OLS Residuals',
            'Inferred Residuals',
            False
        )

        st.plotly_chart(
            fig6,
            use_container_width=True
        )

    st.markdown(
        f"""
        The Sum of the errors can also tell you if the betas were well calculated or not. Normally, when the betas are
        unbiased, the sum of the residuals will be zero (or very close to zero). The Sum of erros are very helpful to 
        calculate the beta biases.
        
        **Sum of OLS Residuals**: {sum(ols_df['error_terms'])}
        
        **Sum of Inferred Residuals**: {sum(ols_df['inferred_error_terms'])} *(Decimals can affect the calculations)*
        """
    )

    st.divider()

    st.title('Hat Matrix')

    st.text(
        """
        The Hat Matrix is very important when calculating the OLS algorith. This Matrix is symetrical and idempotent. 
        When multiplying the Y vector with the Hat Matrix you obtain the Fitted Values. Also you can obtain the 
        Residuals by subtracting the Hat Matrix to the Identity Matrix (zeros and ones in the diagonal) and then
        multiplying to the y Vector.
        """
    )

    col1, col2 = st.columns([3, 7])

    with col1:
        st.write('Formula')
        st.latex(r'''
            \mathbf{H} = \mathbf{X} (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top
            ''')

        st.latex(r'''
        \hat{y} = \mathbf{H} y
        ''')

        st.latex(r'''
        \hat{u} = (\mathbf{I} - \mathbf{H}) y
        ''')

    with col2:
        st.write('Data')
        Some_Matrix = Information_Matrix.dot(X_Variance_Matrix_Inverse)
        Hat_Matrix = Some_Matrix.dot(Information_Matrix_T)

        st.dataframe(
            pd.DataFrame(Hat_Matrix),
            height=175
        )

    st.divider()

    st.title('Correlations and Biases')

    st.text(
        """
        We might also check if there is any correlation between the X's and the Residuals. This might diagnose (or help
        us to diagnose) if there is endogeneity on the regressors.
        
        If the values of the vector are zeros (or near to zeros) we can assume there is endogeneity.
        """
    )

    Y_Hat = Information_Matrix.dot(Beta)
    Residuals_Vector = Y_Vector - Y_Hat
    Intercorrelation_Vector = Information_Matrix_T.dot(Residuals_Vector)

    col1, col2 = st.columns([3, 7])

    with col1:
        st.write('Formula')
        st.latex(r'''
        \mathbf{X}^\top \mathbf{e} = \mathbf{X}^\top (\mathbf{y} - \hat{\mathbf{y}})
        ''')

    with col2:
        st.write('Data')
        st.table(Intercorrelation_Vector)

    st.text(
        """
        The Bias calculation might be interesting for us. Other biases might exist and should be calculated with other 
        ways.
        """
    )

    Bias = X_Variance_Matrix_Inverse.dot(Intercorrelation_Vector)

    col1, col2 = st.columns([3, 7])

    with col1:
        st.write('Formula')
        st.latex(r'''
            \mathbf{S} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{e}
            ''')

    with col2:
        st.write('Data')
        st.table(Bias)

    st.text(
        """
        By calculating the bias, we might understand if our OLS algorithm is working or not:
        """
    )

    b0_slider_3 = st.slider("Select a value for b0", -10.0, 10.0, 6.4294, key='b0_slider_3')

    b1_slider_3 = st.slider("Select a value for b1", -1.0, 1.0, -0.2279, key='b1_slider_3')

    col1, col2 = st.columns([5, 5])

    Fake_Betas = [b0_slider_3, b1_slider_3]
    Fake_Y_Hat = Information_Matrix.dot(Fake_Betas)
    Fake_Residuals_Vector = Y_Vector - Fake_Y_Hat
    Fake_Intercorrelation_Vector = Information_Matrix_T.dot(Fake_Residuals_Vector)
    Fake_Bias = X_Variance_Matrix_Inverse.dot(Fake_Intercorrelation_Vector)

    with col1:
        st.write('Intercorrelation Vector')
        st.table(Fake_Intercorrelation_Vector)

    with col2:
        st.write('Bias')
        st.table(-Fake_Bias)

    st.divider()

    st.title('The Squared Sums and R Squared')

    st.text(
        """
        To evaluate our model we want to calculate the R-squared, for this we need the Squared Sums. The Squared Sum of 
        the Residuals is just that, the Sum of the Squared Residuals:
        """
    )

    SSR = (Residuals_Vector.transpose()).dot(Residuals_Vector)

    col1, col2 = st.columns([5, 5])

    with col1:
        st.write('Formula')
        st.latex(r'''
        SSR = \mathbf{e}^\top \mathbf{e} = \mathbf{y}^\top \mathbf{y} - \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{y}
        ''')

    with col2:
        st.write(f'Data')
        st.metric(label='SSR', value=SSR.round(4))

    st.text(
        """
        The Squared Sum of the Totals (SST) is the variation on the observed phenomenon we want to study:
        """
    )

    SST = (Y_Vector.transpose()).dot(Y_Vector) - (sum(Y_Vector) ** 2) / len(Y_Vector)

    col1, col2 = st.columns([5, 5])

    with col1:
        st.write('Formula')
        st.latex(r'''
        \text{SST} = \mathbf{y}^\top \mathbf{y} - n \cdot \bar{y}^2
        ''')

    with col2:
        st.write(f'Data')
        st.metric(label='SST', value=SST.round(4))

    st.text(
        """
        The Squared Sum of the Estimation (SSE) is the variation of our regression model. That is just the difference 
        between the SST and the SSR.
        """
    )

    SSE = (Beta.transpose()).dot(Y_Covariance_X) - (sum(Y_Vector) ** 2) / len(Y_Vector)

    col1, col2 = st.columns([5, 5])

    with col1:
        st.write('Formula')
        st.latex(r'''
        \text{SSE} = \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{y} - n \cdot \bar{y}^2
        ''')

    with col2:
        st.write(f'Data')
        st.metric(label='SSE', value=SSE.round(4))
