# Neural Evolution

## What?

Most modern neural network architectures are very loosely inspired by real neurons, and from a high level, present
as functions that expect specific kinds of outputs when given specific kinds of inputs.

Brains don't do that. They're much more nebulous and nonlinear in operation. What happens if we try to simulate a
model that behaves much more like real neuron clumps?

## WIP Notes

### Setup
```
conda create --name=neural-evolution python=3.13
pip install -r requirements.txt
```


### Simulation Objectives
1. **Maintain stable neuron population size** ✓

2. Act as a signal wire
3. Act as two independent signal wires
4. Act as four independent signal wires
5. Act as a NOT wire (spam signals except when incoming)
6. 1 wire and 1 NOT wire
7. AND wire

8. Simple memory - input wire and GO wire; progressively more complex signals

9. 2D environment with visual signals for win location
