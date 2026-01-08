
import numpy as np
import matplotlib.pyplot as plt

class Tensor:
    
    """ stores a single scalar Tensor and its gradient """

    def __init__(self, data, _children=(), _op=''):

        self.data = data
        self.grad = 0.0

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):

        # (1) if other is not a Tensor, convert it to one
        other = other if isinstance(other, Tensor) else Tensor(other)

        # (2) create a new Tensor that is the sum of self and other
        out = Tensor(self.data + other.data, (self, other), '+')

        # (3) define the backward function for this operation
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        # (4) return the new Tensor
        return out

    def __mul__(self, other):

        # (1) if other is not a Tensor, convert it to one
        other = other if isinstance(other, Tensor) else Tensor(other)

        # (2) create a new Tensor that is the product of self and other
        out = Tensor(self.data * other.data, [self, other], '*')

        # (3) define the backward function for this operation
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        # (4) return the new Tensor
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data ** other.data, [self, other], '**')

        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += (self.data ** other.data) * (0 if self.data <= 0 else np.log(self.data)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.max(self.data, 0), [self], 'ReLU')

        def _backward():
            self.grad += (1 if self.data >= 0 else 0) * out.grad
        out._backward = _backward

        # (4) return the new Tensor
        return out

    def build_topo(self, visited=None, topo=None):
        if self not in visited:
            visited.add(self)
            for child in self._prev:
                child.build_topo(visited=visited, topo=topo)
            topo.append(self)
        return topo

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        topo = self.build_topo(topo=topo, visited=visited)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


    def __neg__(self): # -self
        return self * Tensor(-1)

    def __radd__(self, other): # other + self
        return other + self

    def __sub__(self, other): # self - other
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward

        return out

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other 

    def __truediv__(self, other): # self / other
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, [self, other], '/')

        def _backward():
            self.grad += 1 / other.data * out.grad
            other.grad += -self.data / (other.data**2) * out.grad
        out._backward = _backward

        return out

    def __rtruediv__(self, other): # other / self
        return other / self

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

def tanh_d(dual_number: Tensor):
    out = Tensor(np.tanh(dual_number.data), (dual_number,), 'tanh')
    def _backward():
        dual_number.grad += (1 - out.data**2) * out.grad
    out._backward = _backward
    return out

# define a neural network, MLP with 1 hidden layer
def func_nn(x, W1, b1, W2, b2):
    h1 = tanh_d(W1*x + b1)
    y = W2*h1 + b2
    return y

def mse_with_backward(y, y_hat):
    loss = (y - y_hat)**2
    loss.backward()
    return loss

def mse(y, y_hat):
    loss = (y - y_hat)**2
    return loss


if __name__ == "__main__":
    
    ## generate data
    np.random.seed(0)
    x = np.linspace(-2, 2, 100)
    y = 2*x + 1 + np.random.randn(100)*0.1

    ## Parameters
    W1 = Tensor(0.1)
    b1 = Tensor(0.2)
    W2 = Tensor(0.3)
    b2 = Tensor(0.4)
    
    lr = 0.01
    nb_epoch = 100
    loss_history = []
    grad_history = []
    for epoch in range(nb_epoch):
        
        lst_loss = []
        grad_sum = 0.0
        for i in range(len(x)):
            x_i = Tensor(x[i])
            y_i = Tensor(y[i])

            y_hat = func_nn(x_i, W1, b1, W2, b2)
            loss = mse(y_i, y_hat)
            loss.backward()

            lst_loss.append(loss.data)
            grad_sum += abs(W1.grad) + abs(b1.grad) + abs(W2.grad) + abs(b2.grad)

            # Print gradients for debugging (first 3 epochs and first 3 samples)
            if epoch < 3 and i < 3:
                print(f"Epoch {epoch+1}, Sample {i+1}")
                print(f"W1.grad: {W1.grad}, b1.grad: {b1.grad}, W2.grad: {W2.grad}, b2.grad: {b2.grad}")
                print(f"W1.data: {W1.data}, b1.data: {b1.data}, W2.data: {W2.data}, b2.data: {b2.data}")
                print(f"loss: {loss.data}")

            # FIXME: Update with the gradient
            W1.data -= lr * W1.grad
            b1.data -= lr * b1.grad
            W2.data -= lr * W2.grad
            b2.data -= lr * b2.grad

            # FIXME: reset gradients
            W1.zero_grad = lambda: None # Helper if needed, but here we just reset .grad
            W1.grad = 0.0
            b1.grad = 0.0
            W2.grad = 0.0
            b2.grad = 0.0

        mean_loss = np.mean(lst_loss)
        mean_grad = grad_sum / len(x)
        loss_history.append(mean_loss)
        grad_history.append(mean_grad)
        print(f"Epoch {epoch+1}/{nb_epoch}, Loss: {mean_loss}")

        # learning rate decay (fix: only decay every 20 epochs)
        if (epoch + 1) % 20 == 0:
            lr *= 0.1

    # Plot loss and gradient
    fig, ax1 = plt.subplots()
    ax1.plot(loss_history, color='tab:blue', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(grad_history, color='tab:red', label='Gradient')
    ax2.set_ylabel('Mean Gradient', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Loss and Mean Gradient per Epoch')
    fig.tight_layout()
    plt.savefig('loss_gradient.png')
    # plt.show() # Commented out to avoid blocking execution in non-interactive environment
