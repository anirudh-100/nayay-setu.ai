"""Drafting journeys: one knowledge module per citizen document the engine can draft.

A *journey* (RTI, consumer complaint, police complaint, …) is defined once here — its
input fields, the official legal facts it cites, how it frames the citizen's situation,
and how it renders the document. The generic ``/draft`` engine (app.services.drafting_
service) serves all of them, so adding a new document type is a single module dropped in
and registered — no new endpoint, schema, or UI code.

Trust rule for every journey: legal scaffolding (statutory sections, forums, fees, time
limits) is hand-authored from the official Act and marked with its source — NEVER
LLM-invented. The model only turns the situation into the document's specific content.
"""
from app.knowledge.drafting.base import (  # noqa: F401
    Journey,
    RenderResult,
    all_journeys,
    coerce_lines,
    get_journey,
    register,
)

# Import journey modules for their registration side effects. Order here = menu order.
from app.knowledge.drafting import rti  # noqa: F401,E402
from app.knowledge.drafting import consumer_complaint  # noqa: F401,E402
from app.knowledge.drafting import police_complaint  # noqa: F401,E402
