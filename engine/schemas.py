"""Pydantic models for every stage's table rows."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

Scenario = Literal["no_climate", "rcp45", "rcp85"]
Hazard   = Literal["heat", "flood", "bushfire"]
Axis     = Literal["capacity", "functionality", "accessibility", "sustainability"]
Prov     = Literal["proxy", "survey"]


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str
    name: str | None = None
    suburb: str | None = None
    council: str
    asset_type: str
    asset_class: str
    component: str
    install_year: int = Field(ge=1800, le=2100)
    useful_life_years: float = Field(gt=0)
    condition: float = Field(ge=1.0, le=5.0)
    grc: float = Field(ge=0.0)
    extent: float | None = None
    lat: float | None = None
    lon: float | None = None
    criticality_seed: float | None = None


class ClimateExposure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str
    scenario: Scenario
    hazard: Hazard
    year: int = Field(ge=2020, le=2100)
    intensity: float = Field(ge=0.0)


class Prior(BaseModel):
    asset_type: str
    component: str
    useful_life_mu: float = Field(gt=0)
    useful_life_sigma: float = Field(gt=0)
    unit_rate: float = Field(gt=0)
    unit_rate_cv: float = Field(ge=0)


class ClimateFactor(BaseModel):
    component: str
    hazard: Hazard
    k: float = Field(ge=0.0)


class CriterionScore(BaseModel):
    asset_id: str
    axis: Axis
    score: float = Field(ge=1.0, le=5.0)
    provenance: Prov


class MCPath(BaseModel):
    realisation: int
    asset_id: str
    scenario: Scenario
    year: int
    condition: float
    renew_need: bool
    cost_if_renewed: float


class OptSolution(BaseModel):
    realisation: int
    asset_id: str
    scenario: Scenario
    renew_year: int | None
    cost: float | None
