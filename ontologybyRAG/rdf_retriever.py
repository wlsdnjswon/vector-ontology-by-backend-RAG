# rdf_retriever.py
# RDF 정보 추출 모듈
import re
from rdflib import Graph, Namespace, URIRef, Literal, FOAF, RDF
from rdflib.plugins.sparql import prepareQuery
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleRdfInfoRetriever:
    """
    질문에서 사람 이름을 찾아 정보 조회 후 구조화된 딕셔너리를 반환합니다.
    """

    def __init__(self, rdf_file_path):
        self.graph = Graph()
        self.NS = Namespace("http://www.example.org/researcher-ontology.owl#") # 온톨로지에 맞게 수정
        self.FOAF = FOAF
        self.RDF = RDF

        try:
            logging.info(f"RDF 파일 로딩 시작: {rdf_file_path}")
            self.graph.parse(rdf_file_path, format="xml")
            logging.info(f"RDF 파일 로딩 완료. 총 트리플 수: {len(self.graph)}")
        except FileNotFoundError:
            logging.error(f"오류: RDF 파일을 찾을 수 없습니다 - {rdf_file_path}")
            raise
        except Exception as e:
            logging.error(f"RDF 파일 로딩 중 오류 발생: {e}")
            raise

        self.graph.bind("ns", self.NS)
        self.graph.bind("foaf", self.FOAF)
        self.graph.bind("rdf", self.RDF)

        # 이름-ID 맵 (실제 환경에서는 동적 생성이 더 좋음)
        self.name_id_map = {
            "정진원": "JungJinWon",
            "이예림": "LeeYelim",
            "신요안": "ShinYoan"
            # 필요에 따라 추가
        }
        self.uri_name_map = self._build_uri_name_map()
        logging.info("이름-URI 맵 및 URI-이름 맵 준비 완료.")

    def _build_uri_name_map(self):
        uri_name_map = {}
        for name, id_val in self.name_id_map.items():
            uri = URIRef(self.NS + id_val)
            uri_name_map[uri] = name

        query = prepareQuery("""
            SELECT ?uri ?name WHERE {
                { ?uri rdf:type ns:Person } UNION { ?uri rdf:type foaf:Person } . # 타입 유연성
                OPTIONAL { ?uri ns:hasName ?name . }
                OPTIONAL { ?uri foaf:name ?name . }
                FILTER(BOUND(?name))
            }
        """, initNs={"ns": self.NS, "foaf": self.FOAF, "rdf": self.RDF})
        try:
            results = self.graph.query(query)
            for row in results:
                if row.uri not in uri_name_map:
                    uri_name_map[row.uri] = str(row.name)
        except Exception as e:
            logging.warning(f"그래프에서 이름 조회 중 오류: {e}")
        return uri_name_map

    def find_person_uri(self, question):
        sorted_names = sorted(self.name_id_map.keys(), key=len, reverse=True)
        for kor_name in sorted_names:
            if kor_name in question:
                person_id = self.name_id_map[kor_name]
                person_uri = URIRef(self.NS + person_id)
                logging.info(f"질문에서 '{kor_name}'(URI: {person_uri}) 발견.")
                return person_uri, kor_name
        logging.warning("질문에서 인식 가능한 사람 이름을 찾지 못했습니다.")
        return None, None

    def get_label_for_node(self, node):
        if isinstance(node, Literal):
            return str(node)
        elif isinstance(node, URIRef):
            if node in self.uri_name_map: return self.uri_name_map[node]
            common_label_props = [self.FOAF.name, self.NS.hasName, URIRef("http://www.w3.org/2000/01/rdf-schema#label")]
            for prop in common_label_props:
                label = self.graph.value(subject=node, predicate=prop)
                if label: return str(label)
            try:
                qname = self.graph.compute_qname(node, generate=False)
                return f"{qname[0]}:{qname[2]}"
            except:
                local_name = node.split('#')[-1].split('/')[-1]
                return local_name if local_name else str(node)
        else:
            return str(node)

    def get_all_related_info(self, person_uri):
        """주어진 사람 URI와 관련된 모든 정보를 구조화된 딕셔너리로 반환합니다."""
        if not person_uri or not isinstance(person_uri, URIRef):
            return {"direct": [], "inverse": []}

        logging.info(f"{person_uri} 관련 정보 조회 시작.")
        related_info = {"direct": [], "inverse": []}
        temp_inverse = {} # 역관계 임시 저장소 {subject_uri: {'label': ..., 'relation': ..., 'details': [...]}}

        # 1. 직접 속성 조회 (person_uri ?p ?o)
        try:
            direct_query_str = """
                SELECT ?p ?displayValue WHERE {
                    {
                        # Case 1: Property is ns:hasSkill
                        BIND(ns:hasSkill AS ?p)
                        ?subject ns:hasSkill ?skillUri .
                        OPTIONAL { ?skillUri ns:hasSkillName ?skillName . }
                        # 스킬 이름이 있으면 사용, 없으면 스킬 URI 사용
                        BIND(COALESCE(?skillName, ?skillUri) AS ?displayValue)
                    } UNION {
                        # Case 2: Property is NOT ns:hasSkill
                        ?subject ?p ?o .
                        # ns:hasSkill과 rdf:type은 Case 1 또는 필터링으로 제외됨
                        FILTER(?p != ns:hasSkill && ?p != rdf:type)
                        BIND(?o AS ?displayValue)
                    }
                }
            """
            direct_query = prepareQuery(direct_query_str, initNs={"ns": self.NS, "rdf": self.RDF})

            results = self.graph.query(direct_query, initBindings={'subject': person_uri})
            logging.info(f"직접 속성 조회 결과 (스킬 처리 포함): {len(results)} 건")
            for row in results:
                predicate_label = self.get_label_for_node(row.p)
                # displayValue는 Literal일 수도 있고, 이름이 없는 스킬 URI 등 다른 URI일 수도 있음
                object_label = self.get_label_for_node(row.displayValue)
                related_info["direct"].append({"predicate": predicate_label, "object": object_label})
        except Exception as e:
            logging.error(f"수정된 직접 속성 조회 중 오류: {e}")

        # 2. 역관계 속성 조회 (?s ?p person_uri) 및 관련 엔티티 상세 정보
        try:
            inverse_query = prepareQuery("""
                SELECT ?s ?p ?s_prop ?s_val WHERE {
                    ?s ?p ?object .
                    OPTIONAL {
                        ?s ?s_prop ?s_val .
                        FILTER(?s_prop != ?p && ?s_prop != rdf:type)
                    }
                } ORDER BY ?s ?p
            """, initNs={"rdf": self.RDF})

            results = self.graph.query(inverse_query, initBindings={'object': person_uri})
            for row in results:
                subject_uri = row.s
                predicate_uri = row.p

                if subject_uri not in temp_inverse:
                     subject_label = self.get_label_for_node(subject_uri)
                     predicate_label = self.get_label_for_node(predicate_uri)
                     temp_inverse[subject_uri] = {
                         'label': subject_label,
                         'relation_to_person': predicate_label,
                         'details': []
                     }

                if row.s_prop and row.s_val:
                    prop_label = self.get_label_for_node(row.s_prop)
                    val_label = self.get_label_for_node(row.s_val)
                    detail_tuple = (prop_label, val_label)
                    # details 리스트 내 중복 방지 확인
                    is_duplicate = False
                    for existing_prop, existing_val in temp_inverse[subject_uri]['details']:
                        if existing_prop == prop_label and existing_val == val_label:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        temp_inverse[subject_uri]['details'].append(detail_tuple)


            related_info["inverse"] = list(temp_inverse.values())
        except Exception as e:
            logging.error(f"역관계 속성 조회 중 오류: {e}")

        logging.info(f"{person_uri} 관련 정보 조회 완료.")
        return related_info