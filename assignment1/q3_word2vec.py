import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N, 1)) + 1e-30

    return x
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print ("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print (x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ("Test passed")

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE
    probabilities = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(probabilities[:,target])
    delta = probabilities
    delta[:,target] -= 1

    N, D = outputVectors.shape

    grad = delta.reshape((N, 1)) * predicted.reshape(1, D)
    gradPred = (delta.reshape((1, N)).dot(outputVectors)).flatten()
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)

#index存入所有U中的编号
    indices = [target]
    #循环Ｋ次，随机生成一个id,存入index.
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]
    #做好标记，帮助后面计算sigmoid与cost.
    labels = np.array([1] + [-1 for k in range(K)])\
    #取出所有index中的矢量．
    vecs = outputVectors[indices, :]
    #乘以输入向量　　Jneg−sample(o,υc,U)　=　−log(σ(μ⊤oυc))−∑k=1Klog(σ(−μ⊤kυc))
    t = sigmoid(vecs.dot(predicted) * labels)
    cost = -np.sum(np.log(t))

    delta = labels * (t - 1)
    # gradPred =(σ(μ⊤oυc)−1)μo−∑k=1K(σ(−μ⊤kυc)−1)μ
    gradPred = delta.reshape((1, K+1)).dot(vecs).flatten()
    #gradtemp      =(σ(μ⊤oυc)−1)υc
    #              =−(σ(−μ⊤kυc)−1)υc,for all k=1,2,…,K
    gradtemp = delta.reshape((K+1, 1)).dot(predicted.reshape((1,
                                                              predicted.shape[0])))



    for k in range(K+1):
        grad[indices[k]] += gradtemp[k, :]

    ### END YOUR CODE
    
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    currentI = tokens[currentWord]
    predicted = inputVectors[currentI, :]

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for cwd in contextWords:
        idx = tokens[cwd]
        cc, gp, gg = word2vecCostAndGradient(predicted, idx, outputVectors,
                                             dataset)
        cost += cc
        gradOut += gg
        gradIn[currentI, :] += gp


    ### END YOUR CODE
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:int(N/2),:]
    outputVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        #centerword为(0,len(sentence))中随机一个word,context为该word左右两侧Ｃ个word,
        # 注意！　len(context)可能小于２C
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:int(N/2), :] += gin / batchsize / denom
        grad[int(N/2):, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print ("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print ("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print ("\n=== Results ===")
    print (skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    print (cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()