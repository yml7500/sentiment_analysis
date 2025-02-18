#! /usr/bin/env python3
import os
from mirascope.core import groq
from pydantic import BaseModel

# Read GROQ API key from file
with open('groq_api_key', 'r') as f:
    os.environ["GROQ_API_KEY"] = f.read().strip()

ARGUMENT_MINER_PROMPT = """
You are analyzing transcript excerpts from a discussion about the Metaverse (online virtual reality spaces).
Your task is to determine if the provided text contains any arguments directly stated by the speaker. Rewrite any argument you find into a single coherent sentence using this structure:
"The speaker argues that [claim] because [reason(s)]."
<criteria>
<argument_definition>
What is an Argument?
An argument is a statement explicitly made by the speaker that:
Expresses a clear position.
Provides a justification or reason supporting this position.
</argument_definition>
<important_notes>
Only consider arguments that the speaker **directly states or clearly implies**. Do not infer beliefs, intentions, or unstated positions.
If the speaker does not explicitly state a position or reason, do not rewrite the text as an argument, even if the position could be inferred.
The rewritten argument must be tied to the exact words and intent of the speaker as presented in the text.
</important_notes>
</criteria>
<output>
Always rewrite the argument as a single sentence following this format:
"The speaker argues that [claim] because [reason(s)]."
If the text does not provide a reason (explicitly or implicitly), do not rewrite it as an argument.
<format>
If the text contains an argument with one or multiple reasons, rewrite it as a single sentence in the following format:
"The speaker argues that [claim] because [reason(s)]."
If the text contains multiple arguments, rewrite each argument as it's own argument sentence in its own xml tag.
<argument>"The speaker argues that [claim] because [reason(s)]."</argument>
<argument>"The speaker argues that [claim] because [reason(s)]."</argument>
If no arguments are present, return the following.:
<argument>None</argument>
Do not include any other text or explanation in your output.
</format>
</output>
<text>
{text}
</text>
"""

PROPOSAL_STANCE_PROMPT = """
Your task is to determine if the provided argument is explicitly or implicitly "for" or "against" the following proposals:
Proposals
vidCaptMember: Video capture should be used in members-only spaces.
vidCaptPublic: Video capture should be used in public spaces.
asdMember: Automatic speech detection should be used in members-only spaces.
asdPublic: Automatic speech detection should be used in public spaces.
Definitions and Criteria
When Does a Statement Relate to a Proposal?
A statement relates to a proposal only if it:
Directly mentions or strongly implies the proposal's key terms.
Fits the context and scope of the proposal.
Addresses the proposal's implementation, justification, or impact.
Output Format
Provide the arguments in the following JSON structure:
{{
  "arguments": [
    {{
      "claim": "proposal_key",
      "stance": "for" or "against"
    }},
    {{
      "claim": "proposal_key",
      "stance": "for" or "against"
    }}
  ]
}}
argument: {argument}
"""


class Argument(BaseModel):
    argument: str
    

class ProposalStance(BaseModel):
    claim: str
    stance: str


class ProposalStances(BaseModel):
    arguments: list[ProposalStance]


@groq.call("llama-3.3-70b-versatile", response_model=Argument, json_mode=True)
def extract_argument(text: str) -> str:
    return ARGUMENT_MINER_PROMPT.format(text=text)


@groq.call("llama-3.3-70b-versatile", response_model=ProposalStances, json_mode=True)
def extract_stance(text: str) -> str:
    return PROPOSAL_STANCE_PROMPT.format(argument=text)


test_inputs = [
    "We need to protect people in public spaces, so I think video capture should be used.",
    "I think we should use video capture to protect people in public spaces.",
    "Today is sunny, so we should go for a hike.",
    "Automatic speech detection helps record keeping, so it should be allowed in public spaces."
]
for text in test_inputs:
    argument = extract_argument(text)
    print(text,argument)
    proposal_stances = extract_stance(argument)
    print(proposal_stances)
    print("-"*100)