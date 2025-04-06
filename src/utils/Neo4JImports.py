import os, logging
from neo4j import GraphDatabase
from retry import retry
from pathlib import Path
from src.config import config
HOSPITALS_CSV_PATH = "/Healthcare/hospitals.csv"
PAYERS_CSV_PATH = "/Healthcare/payers.csv"
PHYSICIANS_CSV_PATH = "/Healthcare/physicians.csv"
PATIENTS_CSV_PATH = "/Healthcare/patients.csv"
VISITS_CSV_PATH = "/Healthcare/visits.csv"
REVIEWS_CSV_PATH = "/Healthcare/reviews.csv"

"""
bolt://svc-neo4j-nodeport:7687
neo4j://svc-neo4j-nodeport:7687
"""
NEO4J_URI = config.NEO4J_URI
NEO4J_USERNAME = config.NEO4J_USERNAME
NEO4J_PASSWORD = config.NEO4J_PASSWORD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

NODES = ["Hospital", "Payer", "Physician", "Patient", "Visit", "Review"]
LOGGER.info(f"Loading {NEO4J_URI}...")

def _set_uniqueness_constraints(tx, node):
    """
    Creates and runs queries enforcing each node to have a unique ID
    """
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node})
        REQUIRE n.id IS UNIQUE;"""
    _ = tx.run(query, {})


@retry(tries=100, delay=10)
def load_hospital_graph_from_csv() -> None:
    """Load structured hospital CSV data following
    a specific ontology into Neo4j"""

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    LOGGER.info("Setting uniqueness constraints on nodes")
    with driver.session(database="neo4j") as session:
        for node in NODES:
            session.execute_write(_set_uniqueness_constraints, node)

    #if Path(HOSPITALS_CSV_PATH).exists() and Path(HOSPITALS_CSV_PATH).is_file():
    LOGGER.info("Loading hospital nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{Path(HOSPITALS_CSV_PATH).as_uri()}' AS hospitals
        MERGE (h:Hospital {{id: toInteger(hospitals.hospital_id),
                            name: hospitals.hospital_name,
                            state_name: hospitals.hospital_state}});
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {HOSPITALS_CSV_PATH}!")

    #if Path(PAYERS_CSV_PATH).exists() and Path(PAYERS_CSV_PATH).is_file():
    LOGGER.info("Loading payer nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{Path(PAYERS_CSV_PATH).as_uri()}' AS payers
        MERGE (p:Payer {{id: toInteger(payers.payer_id),
        name: payers.payer_name}});
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {PAYERS_CSV_PATH}!")

    #if Path(PHYSICIANS_CSV_PATH).exists() and Path(PHYSICIANS_CSV_PATH).is_file():
    LOGGER.info("Loading physician nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{Path(PHYSICIANS_CSV_PATH).as_uri()}' AS physicians
        MERGE (p:Physician {{id: toInteger(physicians.physician_id),
                            name: physicians.physician_name,
                            dob: physicians.physician_dob,
                            grad_year: physicians.physician_grad_year,
                            school: physicians.medical_school,
                            salary: toFloat(physicians.salary)
                            }});
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {PHYSICIANS_CSV_PATH}!")

    #if Path(VISITS_CSV_PATH).exists() and Path(VISITS_CSV_PATH).is_file():
    LOGGER.info("Loading visit nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{Path(VISITS_CSV_PATH).as_uri()}' AS visits
        MERGE (v:Visit {{id: toInteger(visits.visit_id),
                            room_number: toInteger(visits.room_number),
                            admission_type: visits.admission_type,
                            admission_date: visits.date_of_admission,
                            test_results: visits.test_results,
                            status: visits.visit_status
        }})
            ON CREATE SET v.chief_complaint = visits.chief_complaint
            ON MATCH SET v.chief_complaint = visits.chief_complaint
            ON CREATE SET v.treatment_description =
            visits.treatment_description
            ON MATCH SET v.treatment_description = visits.treatment_description
            ON CREATE SET v.diagnosis = visits.primary_diagnosis
            ON MATCH SET v.diagnosis = visits.primary_diagnosis
            ON CREATE SET v.discharge_date = visits.discharge_date
            ON MATCH SET v.discharge_date = visits.discharge_date
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {VISITS_CSV_PATH}!")

    #if Path(PATIENTS_CSV_PATH).exists() and Path(PATIENTS_CSV_PATH).is_file():
    LOGGER.info("Loading patient nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{Path(PATIENTS_CSV_PATH).as_uri()}' AS patients
        MERGE (p:Patient {{id: toInteger(patients.patient_id),
                        name: patients.patient_name,
                        sex: patients.patient_sex,
                        dob: patients.patient_dob,
                        blood_type: patients.patient_blood_type
                        }});
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {PATIENTS_CSV_PATH}!")

    #if Path(REVIEWS_CSV_PATH).exists() and Path(REVIEWS_CSV_PATH).is_file():
    LOGGER.info("Loading review nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{Path(REVIEWS_CSV_PATH).as_uri()}' AS reviews
        MERGE (r:Review {{id: toInteger(reviews.review_id),
                        text: reviews.review,
                        patient_name: reviews.patient_name,
                        physician_name: reviews.physician_name,
                        hospital_name: reviews.hospital_name
                        }});
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {REVIEWS_CSV_PATH}!")

    #if Path(VISITS_CSV_PATH).exists() and Path(VISITS_CSV_PATH).is_file():
    LOGGER.info("Loading 'AT' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{Path(VISITS_CSV_PATH).as_uri()}' AS row
        MATCH (source: `Visit` {{ `id`: toInteger(trim(row.`visit_id`)) }})
        MATCH (target: `Hospital` {{ `id`:
        toInteger(trim(row.`hospital_id`))}})
        MERGE (source)-[r: `AT`]->(target)
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {VISITS_CSV_PATH}!")

    #if Path(REVIEWS_CSV_PATH).exists() and Path(REVIEWS_CSV_PATH).is_file():
    LOGGER.info("Loading 'WRITES' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{Path(REVIEWS_CSV_PATH).as_uri()}' AS reviews
            MATCH (v:Visit {{id: toInteger(reviews.visit_id)}})
            MATCH (r:Review {{id: toInteger(reviews.review_id)}})
            MERGE (v)-[writes:WRITES]->(r)
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {REVIEWS_CSV_PATH}!")

    #if Path(VISITS_CSV_PATH).exists() and Path(VISITS_CSV_PATH).is_file():
    LOGGER.info("Loading 'TREATS' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{Path(VISITS_CSV_PATH).as_uri()}' AS visits
            MATCH (p:Physician {{id: toInteger(visits.physician_id)}})
            MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
            MERGE (p)-[treats:TREATS]->(v)
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {VISITS_CSV_PATH}!")

    #if Path(VISITS_CSV_PATH).exists() and Path(VISITS_CSV_PATH).is_file():
    LOGGER.info("Loading 'COVERED_BY' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{Path(VISITS_CSV_PATH).as_uri()}' AS visits
            MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
            MATCH (p:Payer {{id: toInteger(visits.payer_id)}})
            MERGE (v)-[covered_by:COVERED_BY]->(p)
            ON CREATE SET
                covered_by.service_date = visits.discharge_date,
                covered_by.billing_amount = toFloat(visits.billing_amount)
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {VISITS_CSV_PATH}!")

    #if Path(VISITS_CSV_PATH).exists() and Path(VISITS_CSV_PATH).is_file():
    LOGGER.info("Loading 'HAS' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{Path(VISITS_CSV_PATH).as_uri()}' AS visits
            MATCH (p:Patient {{id: toInteger(visits.patient_id)}})
            MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
            MERGE (p)-[has:HAS]->(v)
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {VISITS_CSV_PATH}!")

    #if Path(VISITS_CSV_PATH).exists() and Path(VISITS_CSV_PATH).is_file():
    LOGGER.info("Loading 'EMPLOYS' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{Path(VISITS_CSV_PATH).as_uri()}' AS visits
            MATCH (h:Hospital {{id: toInteger(visits.hospital_id)}})
            MATCH (p:Physician {{id: toInteger(visits.physician_id)}})
            MERGE (h)-[employs:EMPLOYS]->(p)
        """
        _ = session.run(query, {})
    #else:
    #    LOGGER.error(f"Invalid file {VISITS_CSV_PATH}!")


if __name__ == "__main__":
    load_hospital_graph_from_csv()
