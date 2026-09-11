"""Training callbacks, loaded only when the agent runs with ``--train``.

``setup_training`` is called once after ``callbacks.setup``. Per round, the
engine then calls ``game_events_occurred`` after every step the agent survives
and ``end_of_round`` exactly once at the end. Neither has a time limit.

The rules PDF describes these callbacks loosely; the engine (``environment.py``,
``agents.py``) behaves as follows, and a learner must follow the engine:

- Both states passed to ``game_events_occurred`` carry the *same* step number
  (the step counter is advanced before agents act). Identify a transition by
  ``(round, old_game_state["step"])``.
- On the final step, a *surviving* agent gets ``game_events_occurred`` for it,
  then ``end_of_round`` with the same ``last_game_state``/``last_action`` and
  the same events list plus ``SURVIVED_ROUND``. Do not count that step twice.
- An agent that *died* gets no ``game_events_occurred`` for its death step;
  that transition must be finished in ``end_of_round``. If the round goes on
  (``--continue-without-training``), events from its bombs that explode later,
  including ``KILLED_OPPONENT``, are appended to the same list before
  ``end_of_round``.
- The engine keeps mutating the ``events`` list after the call returns. Copy it
  (``list(events)``) before storing it.

Event names are the string constants in ``events.py`` (``import events as e``).
"""

from .callbacks import Action, AgentSelf, GameState


def setup_training(self: AgentSelf) -> None:
    """Called once, after ``callbacks.setup``, only in training mode.

    Initialise what only training needs: replay buffers, optimisers, counters,
    the pending transition. Declare those attributes on ``AgentSelf``.
    """


def game_events_occurred(
    self: AgentSelf,
    old_game_state: GameState,
    self_action: Action,
    new_game_state: GameState,
    events: list[str],
) -> None:
    """Called after each step the agent survived, including the last one.

    :param old_game_state: The state that was passed to ``act`` this step.
    :param self_action: The action ``act`` returned. A move can still fail
        (``INVALID_ACTION`` in ``events``) when another agent took the target
        tile first under the random action order.
    :param new_game_state: The state after all actions, bombs and explosions
        of this step were resolved.
    :param events: What happened to this agent during the step.
    """


def end_of_round(
    self: AgentSelf,
    last_game_state: GameState,
    last_action: Action,
    events: list[str],
) -> None:
    """Called exactly once per round, whether the agent survived or died.

    A good place to finish the last transition and save the model.

    :param last_game_state: The state passed to ``act`` on the agent's final
        step (the step it died in, or the round's last step).
    :param last_action: The action returned on that step.
    :param events: For a survivor, that final step's events plus
        ``SURVIVED_ROUND`` (the step itself was already reported through
        ``game_events_occurred``). For a dead agent, its death step's events
        plus anything its bombs did afterwards.
    """
