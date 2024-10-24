# I.	Introduction

k-nearest neighbor is one of the simplest (and effective in some cases) supervised-learning algorithms in Machine Learning. When training, this algorithm does not learn anything from the training data (this is also the reason this algorithm is classified as lazy learning), all calculations are performed when it needs to predict the results of new data. K-nearest neighbor can be applied to both types of Supervised learning problems: Classification and Regression. As a classification algorithm, kNN assigns a new data point to the majority set within its neighbors. As a regression algorithm, kNN makes a prediction based on the average of the values closest to the query point. KNN is also known as an Instance-based or Memory-based learning algorithm. kNN is a supervised learning algorithm in which 'k' represents the number of nearest neighbors considered in the classification or regression problem, and 'NN' stands for the nearest neighbors to the number chosen for k.

For example, fruit, vegetable and grain can be distinguished by their crunchiness and sweetness (Figure 1). For the purpose of displaying them on a two-dimension plot, only two characteristics are employed. In reality, there can be any number of predictors, and the example can be extended to incorporate any number of characteristics. In general, fruits are sweeter than vegetables. Grains are neither crunchy nor sweet. Our work is to determine which category does the sweet potato belong to. In this example we choose four nearest kinds of food, they are apple, green bean, lettuce, and corn.

<p align="center">
  <img src="image.png" alt="centered image" width="600px"/>
</p>

<p align="center">
  <b>Image 1:</b> An exmaple of k-nearest neighbor algogirthms.
</p>


There are two important concepts in the above example. One is the method to calculate the distance between sweet potato and other kinds of food. Another concept is the parameter k which decides how many neighbors will be chosen for kNN algorithm. The appropriate choice of k has significant impact on the diagnostic performance of kNN algorithm. A large k reduces the impact of variance caused by random error, but runs the risk of ignoring small but important pattern. The key to choose an appropriate k value is to strike a balance between overfitting and underfitting.

<span style="color:red">

# <span style="color:red"> II.  Calculate similarity</span>

To calculate similarity in the nearest neighbors algorithm, we can choose from the following three options:

## <span style="color:red"> 1. Cosine similarity </span>

Cosine similarity is a mathematical metric used to measure the similarity between two vectors in a multi-dimensional space, particularly in high-dimensional spaces, by calculating the cosine of the angle between them. It follows that the cosine similarity does not depend on the magnitudes of the vectors, but only on their angle. The cosine similarity always belongs to the interval [−1, 1].

<p align="center">
  <img src="image-9.png" alt="centered image" width="600px"/>
</p>

<p align="center">
  <b>Image 2:</b> Examples of cosine similarity.
</p>


The cosine similarity is defined as:

\text{cosine_sim}(u, v) = \frac{
\sum\limits_{i \in I_{uv}} r_{ui} \cdot r_{vi}}
{\sqrt{\sum\limits_{i \in I_{uv}} r_{ui}^2} \cdot
\sqrt{\sum\limits_{i \in I_{uv}} r_{vi}^2}
}

///

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>cosine_sim</mtext>
  <mo stretchy="false">(</mo>
  <mi>u</mi>
  <mo>,</mo>
  <mi>v</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>i</mi>
          <mo>&#x2208;</mo>
          <msub>
            <mi>I</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>u</mi>
              <mi>v</mi>
            </mrow>
          </msub>
        </mrow>
      </munder>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>i</mi>
        </mrow>
      </msub>
      <mo>&#x22C5;</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mi>i</mi>
        </mrow>
      </msub>
    </mrow>
    <mrow>
      <msqrt>
        <munder>
          <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
          <mrow data-mjx-texclass="ORD">
            <mi>i</mi>
            <mo>&#x2208;</mo>
            <msub>
              <mi>I</mi>
              <mrow data-mjx-texclass="ORD">
                <mi>u</mi>
                <mi>v</mi>
              </mrow>
            </msub>
          </mrow>
        </munder>
        <msubsup>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mi>i</mi>
          </mrow>
          <mn>2</mn>
        </msubsup>
      </msqrt>
      <mo>&#x22C5;</mo>
      <msqrt>
        <munder>
          <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
          <mrow data-mjx-texclass="ORD">
            <mi>i</mi>
            <mo>&#x2208;</mo>
            <msub>
              <mi>I</mi>
              <mrow data-mjx-texclass="ORD">
                <mi>u</mi>
                <mi>v</mi>
              </mrow>
            </msub>
          </mrow>
        </munder>
        <msubsup>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>v</mi>
            <mi>i</mi>
          </mrow>
          <mn>2</mn>
        </msubsup>
      </msqrt>
    </mrow>
  </mfrac>
</math>

or

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>cosine_sim</mtext>
  <mo stretchy="false">(</mo>
  <mi>i</mi>
  <mo>,</mo>
  <mi>j</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mo>&#x2208;</mo>
          <msub>
            <mi>U</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>i</mi>
              <mi>j</mi>
            </mrow>
          </msub>
        </mrow>
      </munder>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>i</mi>
        </mrow>
      </msub>
      <mo>&#x22C5;</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>j</mi>
        </mrow>
      </msub>
    </mrow>
    <mrow>
      <msqrt>
        <munder>
          <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mo>&#x2208;</mo>
            <msub>
              <mi>U</mi>
              <mrow data-mjx-texclass="ORD">
                <mi>i</mi>
                <mi>j</mi>
              </mrow>
            </msub>
          </mrow>
        </munder>
        <msubsup>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mi>i</mi>
          </mrow>
          <mn>2</mn>
        </msubsup>
      </msqrt>
      <mo>&#x22C5;</mo>
      <msqrt>
        <munder>
          <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mo>&#x2208;</mo>
            <msub>
              <mi>U</mi>
              <mrow data-mjx-texclass="ORD">
                <mi>i</mi>
                <mi>j</mi>
              </mrow>
            </msub>
          </mrow>
        </munder>
        <msubsup>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mi>j</mi>
          </mrow>
          <mn>2</mn>
        </msubsup>
      </msqrt>
    </mrow>
  </mfrac>
</math>


## <span style="color:red">2. Mean Squared Difference similarity</span>

Mean Squared Difference (MSD) is a statistical method that measures the average of the squared differences between pairs of values. It is commonly used to assess how different two sets of data are from each other. The formula is similar to the Mean Squared Error (MSE), but instead of comparing each value in a set to its expected value, it compares the differences between two sets of values.

<p align="center">
  <img src="image-10.png" alt="centered image" width="600px"/>
</p>

<p align="center">
  <b>Image 3:</b> Examples of Mean Squared Error. A formula that is similar to Mean Squared Difference.
</p>

The Mean Squared Difference is defined as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>msd</mtext>
  <mo stretchy="false">(</mo>
  <mi>u</mi>
  <mo>,</mo>
  <mi>v</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mrow>
      <mo stretchy="false">|</mo>
      <msub>
        <mi>I</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>v</mi>
        </mrow>
      </msub>
      <mo stretchy="false">|</mo>
    </mrow>
  </mfrac>
  <mo>&#x22C5;</mo>
  <munder>
    <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>&#x2208;</mo>
      <msub>
        <mi>I</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>v</mi>
        </mrow>
      </msub>
    </mrow>
  </munder>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>&#x2212;</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>v</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <msup>
    <mo stretchy="false">)</mo>
    <mn>2</mn>
  </msup>
</math>

or

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>msd</mtext>
  <mo stretchy="false">(</mo>
  <mi>i</mi>
  <mo>,</mo>
  <mi>j</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mrow>
      <mo stretchy="false">|</mo>
      <msub>
        <mi>U</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>i</mi>
          <mi>j</mi>
        </mrow>
      </msub>
      <mo stretchy="false">|</mo>
    </mrow>
  </mfrac>
  <mo>&#x22C5;</mo>
  <munder>
    <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mo>&#x2208;</mo>
      <msub>
        <mi>U</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>i</mi>
          <mi>j</mi>
        </mrow>
      </msub>
    </mrow>
  </munder>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>&#x2212;</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>j</mi>
    </mrow>
  </msub>
  <msup>
    <mo stretchy="false">)</mo>
    <mn>2</mn>
  </msup>
</math>

The MSD-similarity is then defined as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable displaystyle="true" columnalign="right left" columnspacing="0em" rowspacing="3pt">
    <mtr>
      <mtd>
        <mtext>msd_sim</mtext>
        <mo stretchy="false">(</mo>
        <mi>u</mi>
        <mo>,</mo>
        <mi>v</mi>
        <mo stretchy="false">)</mo>
      </mtd>
      <mtd>
        <mi></mi>
        <mo>=</mo>
        <mfrac>
          <mn>1</mn>
          <mrow>
            <mtext>msd</mtext>
            <mo stretchy="false">(</mo>
            <mi>u</mi>
            <mo>,</mo>
            <mi>v</mi>
            <mo stretchy="false">)</mo>
            <mo>+</mo>
            <mn>1</mn>
          </mrow>
        </mfrac>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mtext>msd_sim</mtext>
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo>,</mo>
        <mi>j</mi>
        <mo stretchy="false">)</mo>
      </mtd>
      <mtd>
        <mi></mi>
        <mo>=</mo>
        <mfrac>
          <mn>1</mn>
          <mrow>
            <mtext>msd</mtext>
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo>,</mo>
            <mi>j</mi>
            <mo stretchy="false">)</mo>
            <mo>+</mo>
            <mn>1</mn>
          </mrow>
        </mfrac>
      </mtd>
    </mtr>
  </mtable>
</math>

The +1 term is just here to avoid dividing by zero.

## <span style="color:red">3. Pearson correlation coefficient</style>

Pearson Correlation is a statistical method that measures the similarity or correlation between two data objects by comparing their attributes and calculating a score ranging from -1 to +1. A high score indicates high similarity, while a score near zero indicates no correlation. This method is parametric and relies on the mean parameter of the objects, making it more valid for normally distributed data.

<p align="center">
  <img src="image-8.png" alt="centered image" width="600px"/>
</p>

<p align="center">
  <b>Image 4:</b> Examples of scatter diagrams with different values of correlation coefficient (ρ).
</p>

Image: Examples of scatter diagrams with different values of correlation coefficient (ρ).
The Pearson correlation for two objects, with paired attributes, sums the product of their differences from their object means, and divides the sum by the product of the squared differences from the object means, and is defined as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>pearson_sim</mtext>
  <mo stretchy="false">(</mo>
  <mi>u</mi>
  <mo>,</mo>
  <mi>v</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>i</mi>
          <mo>&#x2208;</mo>
          <msub>
            <mi>I</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>u</mi>
              <mi>v</mi>
            </mrow>
          </msub>
        </mrow>
      </munder>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>i</mi>
        </mrow>
      </msub>
      <mo>&#x2212;</mo>
      <msub>
        <mi>&#x3BC;</mi>
        <mi>u</mi>
      </msub>
      <mo stretchy="false">)</mo>
      <mo>&#x22C5;</mo>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mi>i</mi>
        </mrow>
      </msub>
      <mo>&#x2212;</mo>
      <msub>
        <mi>&#x3BC;</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
        </mrow>
      </msub>
      <mo stretchy="false">)</mo>
    </mrow>
    <mrow>
      <msqrt>
        <munder>
          <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
          <mrow data-mjx-texclass="ORD">
            <mi>i</mi>
            <mo>&#x2208;</mo>
            <msub>
              <mi>I</mi>
              <mrow data-mjx-texclass="ORD">
                <mi>u</mi>
                <mi>v</mi>
              </mrow>
            </msub>
          </mrow>
        </munder>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mi>i</mi>
          </mrow>
        </msub>
        <mo>&#x2212;</mo>
        <msub>
          <mi>&#x3BC;</mi>
          <mi>u</mi>
        </msub>
        <msup>
          <mo stretchy="false">)</mo>
          <mn>2</mn>
        </msup>
      </msqrt>
      <mo>&#x22C5;</mo>
      <msqrt>
        <munder>
          <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
          <mrow data-mjx-texclass="ORD">
            <mi>i</mi>
            <mo>&#x2208;</mo>
            <msub>
              <mi>I</mi>
              <mrow data-mjx-texclass="ORD">
                <mi>u</mi>
                <mi>v</mi>
              </mrow>
            </msub>
          </mrow>
        </munder>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>v</mi>
            <mi>i</mi>
          </mrow>
        </msub>
        <mo>&#x2212;</mo>
        <msub>
          <mi>&#x3BC;</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>v</mi>
          </mrow>
        </msub>
        <msup>
          <mo stretchy="false">)</mo>
          <mn>2</mn>
        </msup>
      </msqrt>
    </mrow>
  </mfrac>
</math>

or

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>pearson_sim</mtext>
  <mo stretchy="false">(</mo>
  <mi>i</mi>
  <mo>,</mo>
  <mi>j</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mo>&#x2208;</mo>
          <msub>
            <mi>U</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>i</mi>
              <mi>j</mi>
            </mrow>
          </msub>
        </mrow>
      </munder>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>i</mi>
        </mrow>
      </msub>
      <mo>&#x2212;</mo>
      <msub>
        <mi>&#x3BC;</mi>
        <mi>i</mi>
      </msub>
      <mo stretchy="false">)</mo>
      <mo>&#x22C5;</mo>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>j</mi>
        </mrow>
      </msub>
      <mo>&#x2212;</mo>
      <msub>
        <mi>&#x3BC;</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
        </mrow>
      </msub>
      <mo stretchy="false">)</mo>
    </mrow>
    <mrow>
      <msqrt>
        <munder>
          <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mo>&#x2208;</mo>
            <msub>
              <mi>U</mi>
              <mrow data-mjx-texclass="ORD">
                <mi>i</mi>
                <mi>j</mi>
              </mrow>
            </msub>
          </mrow>
        </munder>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mi>i</mi>
          </mrow>
        </msub>
        <mo>&#x2212;</mo>
        <msub>
          <mi>&#x3BC;</mi>
          <mi>i</mi>
        </msub>
        <msup>
          <mo stretchy="false">)</mo>
          <mn>2</mn>
        </msup>
      </msqrt>
      <mo>&#x22C5;</mo>
      <msqrt>
        <munder>
          <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mo>&#x2208;</mo>
            <msub>
              <mi>U</mi>
              <mrow data-mjx-texclass="ORD">
                <mi>i</mi>
                <mi>j</mi>
              </mrow>
            </msub>
          </mrow>
        </munder>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mi>j</mi>
          </mrow>
        </msub>
        <mo>&#x2212;</mo>
        <msub>
          <mi>&#x3BC;</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>j</mi>
          </mrow>
        </msub>
        <msup>
          <mo stretchy="false">)</mo>
          <mn>2</mn>
        </msup>
      </msqrt>
    </mrow>
  </mfrac>
</math>

</span>

<span style="color:red">
Note: if there are no common users or items, similarity will be 0 (and not -1).
</span>

<span style="color: blue">

# <span style="color:blue"> III. User-based & Item-based </span>

## <span style="color:blue">1. Used-based Classification</span>
The vote v<sub>ij</sub> given by k-NN of u for the rating r &isin; S can be obtained as the sum of the similarity weights of neighbors that have given this rating to i:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>v</mi>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>r</mi>
    </mrow>
  </msub>
  <mo>=</mo>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
            <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>i</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>u</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>&#948</mtext>
      <mo stretchy="false">(</mo>
      <msubsup>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>v</mi>
            <mi>i</mi>
          </mrow>
        </msubsup>
      <mo>=</mo>
      <mi>r</mi>
      <mo stretchy="false">)</mo>
      <mo>&#x22C5;</mo>
      <msub>
        <mi>w</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>v</mi>
        </mrow>
      </msub>
    </mrow>
    
</math>
where &#948(r<sub>vi</sub> = r) is 1 if r<sub>vi</sub> =r, and 0 otherwise.Once this has been computed for every possible rating value, the predicted rating is simply the value r for which v<sub>ir</sub> is the greatest.

A classification method that considers normalized ratings can also be defined. Let S′ be the set of possible normalized values (that may require discretization), the predicted rating is obtained as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"> 
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <mrow>
    <mi>h</mi>
    <mo>⁻¹</mo>
    <!-- Dấu ngoặc ngoài cùng lớn -->
    <mo stretchy="true">(</mo>
    <!-- Biểu thức argmax -->
    <munder>
      <mo data-mjx-texclass="OP" movablelimits="false">argmax</mo>
      <!-- Điều kiện r ∈ S' -->
      <mrow>
        <mi>r</mi>
        <mo>&#x2208;</mo>
        <msup>
          <mi>S'</mi>
        </msup>
      </mrow>
    </munder>
    <mrow>
      <!-- Tổng với điều kiện dưới chân sigma -->
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>i</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>u</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <!-- Biểu thức trong tổng -->
       <mtext>&#948</mtext>
      <mo stretchy="true">(</mo>
      <mi>h</mi>
      <mo stretchy="false">(</mo>
      <msubsup>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>v</mi>
            <mi>i</mi>
          </mrow>
        </msubsup>
      <mo stretchy="false">)</mo>
      <mo>=</mo>
      <mi>r</mi>
      <mo stretchy="true">)</mo>
      <mo>&#x22C5;</mo>
      <msub>
        <mi>w</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>v</mi>
        </mrow>
      </msub>
    </mrow>
    <!-- Dấu ngoặc ngoài cùng lớn -->
    <mo stretchy="true">)</mo>
  </mrow>
</math>

## <span style="color: blue">2. Item-based Classification </span>
This idea can be formalized as follows. Denote N<sub>u</sub>(i) the items rated by user u most similar to item i. The predicted rating of u for i is obtained as a weighted average of the ratings given by u to the items of N<sub>u</sub>(i):

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>u</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <msub>
        <mi>w</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>i</mi>
          <mi>j</mi>
        </mrow>
      </msub>
      <mo>&#x22C5;</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>j</mi>
        </mrow>
      </msub>
    </mrow>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>u</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <msub>
        <mo stretchy="false">|</mo>
        <mi>w</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>i</mi>
          <mi>j</mi>
        </mrow>
        <mo stretchy="false">|</mo>
      </msub>
    </mrow>
  </mfrac>
</math>
Again, the differences in the users’ individual rating scales can be considered by
normalizing ratings with a h:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
<msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <mrow>
    <mi>h</mi>
    <mo>⁻¹</mo>
    <mo stretchy="true">(</mo>
    <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>u</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <msub>
        <mi>w</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>i</mi>
          <mi>j</mi>
        </mrow>
      </msub>
      <mo>&#x22C5;</mo>
      <mi>h</mi>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>j</mi>
        </mrow>
      </msub>
      <mo stretchy="false">)</mo>
    </mrow>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>u</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <msub>
        <mo stretchy="false">|</mo>
        <mi>w</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>i</mi>
          <mi>j</mi>
        </mrow>
        <mo stretchy="false">|</mo>
      </msub>
    </mrow>
  </mfrac>
  <mo stretchy="true">)</mo>
  </mrow>

</math>

Moreover, we can also define an item-based classification approach. In this case,
the items j rated by user u vote for the rating to be given to a new item i, and these votes are weighted by the similarity between i and j. The normalized version of this approach can be expressed as follows:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"> 
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <mrow>
    <mi>h</mi>
    <mo>⁻¹</mo>
    <!-- Dấu ngoặc ngoài cùng lớn -->
    <mo stretchy="true">(</mo>
    <!-- Biểu thức argmax -->
    <munder>
      <mo data-mjx-texclass="OP" movablelimits="false">argmax</mo>
      <!-- Điều kiện r ∈ S' -->
      <mrow>
        <mi>r</mi>
        <mo>&#x2208;</mo>
        <msup>
          <mi>S'</mi>
        </msup>
      </mrow>
    </munder>
    <mrow>
      <!-- Tổng với điều kiện dưới chân sigma -->
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>u</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <!-- Biểu thức trong tổng -->
       <mtext>&#948</mtext>
      <mo stretchy="true">(</mo>
      <mi>h</mi>
      <mo stretchy="false">(</mo>
      <msubsup>
          <mi>r</mi>
          <mrow data-mjx-texclass="ORD">
            <mi>u</mi>
            <mi>j</mi>
          </mrow>
        </msubsup>
      <mo stretchy="false">)</mo>
      <mo>=</mo>
      <mi>r</mi>
      <mo stretchy="true">)</mo>
      <mo>&#x22C5;</mo>
      <msub>
        <mi>w</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>i</mi>
          <mi>j</mi>
        </mrow>
      </msub>
    </mrow>
    <!-- Dấu ngoặc ngoài cùng lớn -->
    <mo stretchy="true">)</mo>
  </mrow>
</math>
</span>

<span style="color: green">

# <span style="color:green">IV. Normalization </span>

These are algorithms that are directly derived from a basic nearest neighbors approach. The notations sim(u,v) for users and sim(i,j) for items result from one of the three similarity calculation options mentioned above.

## <span style="color:green"> 1. Basic k-NN Algorithm </span>

A basic collaborative filtering algorithm.

The prediction 

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
</math> is set as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>i</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>u</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>u</mi>
      <mo>,</mo>
      <mi>v</mi>
      <mo stretchy="false">)</mo>
      <mo>&#x22C5;</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mi>i</mi>
        </mrow>
      </msub>
    </mrow>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>i</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>u</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>u</mi>
      <mo>,</mo>
      <mi>v</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </mfrac>
</math>

or

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>u</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo>,</mo>
      <mi>j</mi>
      <mo stretchy="false">)</mo>
      <mo>&#x22C5;</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>j</mi>
        </mrow>
      </msub>
    </mrow>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>u</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo>,</mo>
      <mi>j</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </mfrac>
</math>

## <span style="color:green">2. k-NN Algorithm with mean</span>

A basic collaborative filtering algorithm, taking into account the mean ratings of each user.

The prediction 

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
</math> is set as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <msub>
    <mi>&#x3BC;</mi>
    <mi>u</mi>
  </msub>
  <mo>+</mo>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>i</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>u</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>u</mi>
      <mo>,</mo>
      <mi>v</mi>
      <mo stretchy="false">)</mo>
      <mo>&#x22C5;</mo>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mi>i</mi>
        </mrow>
      </msub>
      <mo>&#x2212;</mo>
      <msub>
        <mi>&#x3BC;</mi>
        <mi>v</mi>
      </msub>
      <mo stretchy="false">)</mo>
    </mrow>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>i</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>u</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>u</mi>
      <mo>,</mo>
      <mi>v</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </mfrac>
</math>

## <span style="color:green">3. k-NN Algorithm with z-score</span>

A basic collaborative filtering algorithm, taking into account the z-score normalization of each user.

The prediction 

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
</math> is set as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <msub>
    <mi>&#x3BC;</mi>
    <mi>u</mi>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>&#x3C3;</mi>
    <mi>u</mi>
  </msub>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>i</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>u</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>u</mi>
      <mo>,</mo>
      <mi>v</mi>
      <mo stretchy="false">)</mo>
      <mo>&#x22C5;</mo>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mi>i</mi>
        </mrow>
      </msub>
      <mo>&#x2212;</mo>
      <msub>
        <mi>&#x3BC;</mi>
        <mi>v</mi>
      </msub>
      <mo stretchy="false">)</mo>
      <mrow data-mjx-texclass="ORD">
        <mo>/</mo>
      </mrow>
      <msub>
        <mi>&#x3C3;</mi>
        <mi>v</mi>
      </msub>
    </mrow>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>v</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>i</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>u</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>u</mi>
      <mo>,</mo>
      <mi>v</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </mfrac>
</math>

or

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>r</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>u</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <msub>
    <mi>&#x3BC;</mi>
    <mi>i</mi>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>&#x3C3;</mi>
    <mi>i</mi>
  </msub>
  <mfrac>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>u</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo>,</mo>
      <mi>j</mi>
      <mo stretchy="false">)</mo>
      <mo>&#x22C5;</mo>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>r</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>u</mi>
          <mi>j</mi>
        </mrow>
      </msub>
      <mo>&#x2212;</mo>
      <msub>
        <mi>&#x3BC;</mi>
        <mi>j</mi>
      </msub>
      <mo stretchy="false">)</mo>
      <mrow data-mjx-texclass="ORD">
        <mo>/</mo>
      </mrow>
      <msub>
        <mi>&#x3C3;</mi>
        <mi>j</mi>
      </msub>
    </mrow>
    <mrow>
      <munder>
        <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
        <mrow data-mjx-texclass="ORD">
          <mi>j</mi>
          <mo>&#x2208;</mo>
          <msubsup>
            <mi>N</mi>
            <mi>u</mi>
            <mi>k</mi>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </munder>
      <mtext>sim</mtext>
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo>,</mo>
      <mi>j</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </mfrac>
</math>

</span>



# V.	Evaluation
## 1. Advantages:

**Easy to implement** as the complexity of the algorithm is not that high.

**Adapts Easily** – As per the working of the KNN algorithm it stores all the data in memory storage and hence whenever a new example or data point is added then the algorithm adjusts itself as per that new example and has its contribution to the future predictions as well.

**Few Hyperparameters** – The only parameters which are required in the training of a KNN algorithm are the value of k and the choice of the distance metric which we would like to choose from our evaluation metric.


## 2. Disadvantages:  

**Does not scale** – As we have heard about this that the KNN algorithm is also considered a Lazy Algorithm. The main significance of this term is that this takes lots of computing power as well as data storage. This makes this algorithm both time-consuming and resource exhausting.

**Curse of Dimensionality** – There is a term known as the peaking phenomenon according to this the KNN algorithm is affected by the curse of dimensionality which implies the algorithm faces a hard time classifying the data points properly when the dimensionality is too high.

**Prone to Overfitting** – As the algorithm is affected due to the curse of dimensionality it is prone to the problem of overfitting as well. Hence generally feature selection as well as dimensionality reduction techniques are applied to deal with this problem.

# VI. REFERENCES

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4916348/

https://www.geeksforgeeks.org/k-nearest-neighbours/

https://medium.com/@rizwanayasmeen06/k-nearest-neighbor-knn-algorithm-in-machine-learning-d38d9638d7e0

https://www.datastax.com/guides/what-is-cosine-similarity

https://statisticsbyjim.com/regression/mean-squared-error-mse/

https://en.wikipedia.org/wiki/Root_mean_square_deviation

https://www.sciencedirect.com/topics/computer-science/pearson-correlation
