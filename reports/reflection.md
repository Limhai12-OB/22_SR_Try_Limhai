# Reflection draft

*Review and personalize this draft so it reflects your own learning experience.*

The local Naive RAG pipeline worked well at finding documents related to direct
questions. In the recorded tests, the printer, VPN, backup, and wireless-network
questions each retrieved chunks from the expected file. The five active documents
produced twenty chunks, and Chroma kept the index between runs. Separating loading,
chunking, embedding, retrieval, and generation also made the data flow easier to
inspect. The terminal option to show chunks helped connect each answer to its
evidence.

The harder part was obtaining complete, consistently cited answers after retrieval
had already succeeded. The backup answer omitted verification and scheduling,
even though those steps were present in the retrieved material. The VPN answer
used a numbered reference instead of the requested filename. These results show
that retrieving the correct document does not guarantee a complete answer.
Fixed-size chunks can also split sentences and instructions at awkward positions.

The vacation-policy question demonstrated grounding clearly. Retrieval still
returned nearby vectors, but the model said it did not have enough information
instead of inventing a company policy. This is one successful test, not proof that
the model will always refuse unsupported questions.

For a future Advanced RAG improvement, I would try heading-aware chunking that
keeps each numbered procedure together. I would compare it against the current
800-character chunks using the same questions, checking answer completeness,
source accuracy, and whether unsupported questions still receive appropriate
refusals.
