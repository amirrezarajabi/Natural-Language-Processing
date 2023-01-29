# Natural Language Processing with Classification and Vector-Spaces

## Supervised ML

*   `X` Features
*   `Y` Labels
*   prediction function , $\theta$
*   repeat:
    *   `X` $\rightarrow$ prediction function ($\theta, X$) $\rightarrow \hat{Y}$ 
    *   Cost $\hat{Y}$ vs $Y$ $\xrightarrow{\text{minimize}}$ update $\theta$

### Sentiment in Tweets

#### Using Sparse representation
*   Tweets = $[\text{tweet}_1, \text{tweet}_2, \text{... }, \text{tweet}_n]$
*   Vocabluaries creation $\bold{V} = [\text{v}_1, \text{v}_2, \text{... }, \text{v}_{|V|}]$
*   represent each tweet using $\bold{\text{Sparse representation}}$
*   I am happy $\longrightarrow [\text{0, 0, ..., 0, 0},\text{1},\text{0, 0, ..., 0, 0},\text{1},\text{0, 0, ..., 0, 0},\text{1},\text{0, 0, ..., 0, 0}]$
*   $\theta = [\theta_0, \theta_1, \text{... }, \theta_{|V|}]\text{ } \xrightarrow{\text{problems}}$ $\begin{cases} \text{large training time} \\ \text{large prediction time} \end{cases}$
#### Using frequency
*   Tweets = $[\text{tweet}_1, \text{tweet}_2, \text{... }, \text{tweet}_n]$
*   Vocabluaries creation $\bold{V} = [\text{v}_1, \text{v}_2, \text{... }, \text{v}_{|V|}]$
* create `freqs` map $(\text{word, class}) \longrightarrow to frequency$
*   $\text{tweet}_{i} \longrightarrow X_i = [1, \sum_w \text{freqs}(w, +),  \sum_w \text{freqs}(w, -)]$
*   $\theta = [\theta_0, \theta_1, \theta_2]$

#### preprocessing on tweets
*   stop words and punctuation
*   Handles and URLs
*   stemming and lowercasing

## Bayes's rule
*   $P(X|Y) = \LARGE{\frac{P(X \bigcap Y)}{P(Y)}}$

*   $P(X|Y) = \LARGE{\frac{P(Y|X)P(X)}{P(Y)}}$

### Sentiment in tweets
*   create `freqs` map $(\text{word, class}) \longrightarrow to frequency$
*   $N_{class}$ sum over words in class
*   $P(\text{word}|\text{class}) = \LARGE{\frac{\text{freqs}(\text{word, class})}{N_{class}}}$
*   Laplacian smoothing
    *   $|V_{\text{class}}|$ is length of uniqe words in vocabluary
    *   $P(\text{word}|\text{class}) = \LARGE{\frac{\text{freqs}(\text{word, class}) + 1}{N_{class}+|V_{\text{class}}|}}$ 
*   $\LARGE{\text{ratio}_{\text{word}}} = \LARGE{\frac{P(\text{word}|\text{+})}{P(\text{word}|\text{--})}}$
*   $\begin{cases}\large{\frac{P(\text{+})}{P(\text{--})}\prod_{\text{w}}^{\text{tweet}}\text{ratio}_{\text{w}} > 1} & + \\  \text{O.W} & -\end{cases}$
*   ### Log Likelihood
    *   products bring risk of underflow
    *   $log(a*b) = log(a) \text{ + } log(b)$
    *   $log(\frac{P(+)}{P(-)}) + \sum_{\text{w}}^{\text{tweet}}log(\text{ratio}_{\text{w}})$
    *   $\lambda_{\text{w}} =  log(\text{ratio}_{\text{w}})$
    *   $\begin{cases}\large{log(\frac{P(+)}{P(-)}) + \sum_{\text{w}}^{\text{tweet}}log(\text{ratio}_{\text{w}}) > 0} & + \\  \text{O.W} & -\end{cases}$


