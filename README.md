# Topology_Superposition_Theorem

This is a package for efficient combinatorial topological actions power flow computation based on the extended superposition theorem for power systems.

Here is the extended superposition theroem for topological changes. The resulting powerflows is a linear combination of unitary change power flows:

𝑃𝐹(𝑇)=𝛼×𝑃𝐹(𝑇𝑟𝑒𝑓)+𝛽1×𝑃𝐹(𝑇𝑟𝑒𝑓∘𝜏1)+𝛽2×𝑃𝐹(𝑇𝑟𝑒𝑓∘𝜏2)

with 𝑇=𝑇𝑟𝑒𝑓∘𝜏1∘𝜏2 and 𝛼=1−𝛽1−𝛽2

We have 𝑇𝑟𝑒𝑓 as the reference topology from which we apply topological changes 𝜏1 and 𝜏2 in indifferent order to reach a target topology 𝑇. Finding the betas simply stems from solving a linear system of dimension the number of considered changes. Only minimal information from individual power flow state is needed for this, without knowledge of any underlying grid properties or complete adjacency matrix.

For more information, see paper (under writing) and abstract in reference folder.

# Get started

you can install the package from pypi
```
pip install topologysuperpositiontheorem
```

you can them run the getting started notebook to get familiar with the package